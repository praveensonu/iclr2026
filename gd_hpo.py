import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import LoraConfig, get_peft_model 
from data_module import DualDatasetCarte
from collators import custom_gd_collator_forget, custom_data_collator_forget
from utils import find_all_linear_names
from forget_trainer import GradDiffTrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE
from eval_utils import compute_fe_scores, compute_mu_scores
from tabulate import tabulate
from utils import update_json_dict, coerce_to_list
import os
import shutil
from accelerate.utils import broadcast_object_list
import gc

accelerator = Accelerator()

cfg = Config() #change this based on dataset

# ------- loading the datafiles-------------

print('loading the forget, retain')
forget = pd.read_csv(cfg.forget_path, converters={'perturbed_answers': coerce_to_list}) 
if cfg.retain_path.endswith('.csv'):
    retain = pd.read_csv(cfg.retain_path)
elif cfg.retain_path.endswith('.json'):
    retain = pd.read_json(cfg.retain_path)
elif cfg.retain_path.endswith('.parquet'):
    retain = pd.read_parquet(cfg.retain_path)

retain_full = pd.read_csv(cfg.retain_full)
print('forget shape:', forget.shape)
print('retain shape:', retain.shape)


# ------- Load the tokenizer ----------------
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token

# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    return df

forget = make_template_format(forget)
retain = make_template_format(retain)
retain_full = make_template_format(retain_full)
print('forget question and answer\n',forget['question'][0], forget['answer'][0])
print('\n\nretain question and answer\n',retain['question'][0], retain['answer'][0])


# ------- creating the dataset ----------------
print('\n\ncreating the dataset for cyclic. If you didnt choose this stop the training now')
dataset = DualDatasetCarte(forget_data = forget,
                    retain_data = retain,
                    tokenizer = tokenizer,
                    max_length = 256)
print('\nlength of the dataset',len(dataset))

target_fe_score = 0.90
num_epochs = [5,10,15,20,25,30] ##need to change this line

temp_training_dir = os.path.join(cfg.save_dir, 'tmp_training_output')

final_model_path = os.path.join(cfg.save_dir, 'final_model')

found_model = False

for epoch in num_epochs:
    accelerator.print(f'starting training for {epoch} epochs')
    accelerator.print(f'Loading fresh model: {cfg.model_id}')
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,)
    config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = find_all_linear_names(model),
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

    print(f"{config.target_modules}")

# ------- wrapping the model with the LoRA configuration
    model = get_peft_model(model, config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir = temp_training_dir,
        overwrite_output_dir= True,
        learning_rate = cfg.lr,
        per_device_train_batch_size= cfg.batch_size, 
        num_train_epochs= epoch,
        weight_decay = cfg.weight_decay,
        logging_dir = f'{temp_training_dir}/logs',
        logging_steps = 100,
        eval_strategy= 'no',
        label_names = ['labels'],
        bf16 = True,
        gradient_accumulation_steps= cfg.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        report_to = "none",
    )

    trainer = GradDiffTrainer(
            model = model,
            args = training_args,
            train_dataset = dataset,
            tokenizer = tokenizer,
            data_collator = custom_gd_collator_forget,
            )

    trainer.train()
    accelerator.wait_for_everyone()

    model.eval()
    device = model.device
    
    fe_container = [0.0]

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.eval()
        device = unwrapped_model.device

        accelerator.print("\n-- Evaluating Forget Efficacy (FE) score ---")

        _,_,fe,_ = compute_fe_scores(forget,unwrapped_model, tokenizer, device)
        fe_container[0] = float(fe)
        accelerator.print(f"FE Score after {epoch} epochs: {fe_container[0]:.4f}")
    
    broadcast_object_list(fe_container)
    accelerator.wait_for_everyone()

    current_fe = fe_container[0]

    # fe_value_tensor = torch.tensor(current_fe, device=accelerator.device)
    # fe_value_tensor = accelerator.broadcast(fe_value_tensor)
    # current_fe = fe_value_tensor.item()

    if current_fe >= target_fe_score:
        accelerator.print(f"\nTarget FE score of {target_fe_score} reached with {epoch} epochs!")
        final_model_to_eval  = accelerator.unwrap_model(trainer.model)

        accelerator.save_model(final_model_to_eval, final_model_path)

        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(final_model_path)
            accelerator.print(f"Final model and tokenizer saved to: {final_model_path}")

        # ----- Since we found the model, lets evaluate
        if accelerator.is_main_process:
            accelerator.print("\n- Running Final, Full evaluation on the best model ---")
            eval_device = final_model_to_eval.device
            with torch.no_grad():
                df_f, all_scores_f, fe, f_ppl = compute_fe_scores(
                    df = forget,
                    model = final_model_to_eval,
                    tokenizer = tokenizer,
                    device = eval_device,
                )
                df_r, all_scores_r, mu, rt_ppl = compute_mu_scores(
                    df = retain_full,
                    model = final_model_to_eval,
                    tokenizer = tokenizer,
                    device = eval_device,
                )
            best_params = {
                "num_epochs": epoch,
                "learning_rate": cfg.lr,
            }

            forget_eval = df_f.copy()
            forget_eval['type'] = 'forget'
            retain_eval = df_r.copy()
            retain_eval['perturbed_answers'] = None

            df_out = pd.concat([forget_eval, retain_eval], axis=0)
            out_csv = f'./results/datasets/{cfg.ds_type}/{cfg.loss_type}_{cfg.exp_type}.csv'
            df_out.to_csv(out_csv, index=False)
            metrics = [
                ("FE", float(fe)),
                ("MU", float(mu)),
                ("PPL-F", float(getattr(f_ppl, "item", lambda: f_ppl)())),
                ("PPL-R", float(getattr(rt_ppl, "item", lambda: rt_ppl)())),
            ]
            print("\n\n============ FINAL RESULTS ============\n")
            print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))
            
            # --- Save metrics JSON ---
            
            results = {f'{cfg.loss_type}_{cfg.exp_type}': 
                        {"FE":      fe.item(),
                        "MU":   mu.item(),
                        'forget_scores' : all_scores_f.tolist(),
                        'retain_scores': all_scores_r.tolist(),
                        "PPL-F" : f_ppl.item(),
                        "PPL-R":  rt_ppl.item(),
                        'model_id': cfg.model_id,
                        'batch_size': cfg.batch_size * 2 * cfg.gradient_accumulation_steps ,
                        'num_epochs': int(best_params["num_epochs"]),
                        'lr': cfg.lr,
                        'weight_decay': cfg.weight_decay,
                        'LoRA_r': cfg.LoRA_r,
                        'LoRA_alpha': cfg.LoRA_alpha,
                        }}
            result_json_path = f'./results/scores/wpu/random/{cfg.loss_type}_{cfg.exp_type}_results.json'
            update_json_dict(f'./results/scores/wpu/random/{cfg.loss_type}_{cfg.exp_type}_results.json', results)
            accelerator.print(f"Final metrics saved to {result_json_path}")
        break
    else:
        accelerator.print(f"FE score of {current_fe} is less than the target score of {target_fe_score}. Training for more epochs...")

    accelerator.print("Cleaning up memory for the next run...")
    del model
    del trainer

    if accelerator.is_main_process:
        del unwrapped_model

    gc.collect()

    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


accelerator.wait_for_everyone()
if accelerator.is_main_process:
    if found_model:
        accelerator.print("\nScript finished successfully.")
        accelerator.print(f"Removing temporary directory: {temp_training_dir}")
        shutil.rmtree(temp_training_dir)
    else:
        accelerator.print("\nFinished all epoch trials, but the target FE score was not reached.")


