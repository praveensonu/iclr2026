import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

print("right now using device num:", os.environ['CUDA_VISIBLE_DEVICES'])
import pandas as pd
from eval_utils import compute_mu_scores, compute_fe_scores
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from peft import PeftModel
from utils import update_json_dict, coerce_to_list
from template import LLAMA3_CHAT_TEMPLATE
import warnings
from transformers import logging as hf_logging
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

hf_logging.set_verbosity_error()

warnings.filterwarnings("ignore")

cfg = Config()
print('loading forget and test set')

forget = pd.read_csv(cfg.forget_path, converters={'perturbed_answers': coerce_to_list})
retain_data = pd.read_csv(cfg.retain_full)

device = 'cuda'


# ---- Loading Tokenizer -----------
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = '<|finetune_right_pad_id|>'



print(f'\n\nConducting evaluation on: {cfg.loss_type}_{cfg.exp_type}')

# ---- Loading model -----------
if cfg.loss_type == 'pre_unlearning':
     model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
else:
     print('loading peft model from ', cfg.save_dir)
     base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id, token = cfg.access_token, device_map = "auto", torch_dtype=torch.bfloat16)
     model = PeftModel.from_pretrained(base_model, cfg.save_dir, device_map="auto", torch_dtype=torch.bfloat16) 
     model = model.merge_and_unload()


# ------- creating template format for tokenization --------
def make_template_format(df):
     df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
     return df

forget_2 = make_template_format(forget)
retain_data = make_template_format(retain_data)
forget_2 = forget_2[['title','question', 'answer', 'perturbed_answers','paraphrased_question', 'paraphrased_answer', 'num_tokens']]
print(forget_2.columns)


model_name = cfg.retriever_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer(model_name, device=device)

print('\ncalculating forget efficacy')

forget,all_scores_fe, fe, f_ppl  = compute_fe_scores(df=forget_2, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device)
print('\nforget efficacy', fe)


print('\ncalculating model utility')

retain,all_scores_mu, mu, rt_ppl  = compute_mu_scores(df=retain_data, model=model, tokenizer=tokenizer, embedding_model=embedding_model, device=device)

forget['type'] = 'forget'
retain['perturbed_answers'] = None
retain['truth'] = None

df = pd.concat([forget, retain], axis=0)
df.to_csv(f'./results/datasets/{cfg.loss_type}_{cfg.exp_type}.csv', index = False) 

metrics = [
    ("FE",      fe.item()),
    ("MU",   mu.item()),
    ("PPL-F",   f_ppl.item()),
    ("PPL-R",rt_ppl.item()),
]

try:
    from tabulate import tabulate
    print("\n\n============ ALL RESULTS ============\n")
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="github"))
    
except ImportError:
    col1_w = max(len(name) for name, _ in metrics)
    col2_w = max(len(f"{val:.4f}") for _, val in metrics)

    print("\n\n============ ALL RESULTS ============\n")
    print(f"| {'Metric'.ljust(col1_w)} | {'Value'.rjust(col2_w)} |")
    print(f"|{'-'*(col1_w+2)}|{'-'*(col2_w+2)}|")
    for name, val in metrics:
        print(f"| {name.ljust(col1_w)} | {val:>{col2_w}.4f} |")


results = {f'{cfg.loss_type}_{cfg.exp_type}': 
            {"FE":      fe.item(),
            "MU":   mu.item(),
            'forget_scores' : all_scores_fe.tolist(),
            'retain_scores': all_scores_mu.tolist(),
            "PPL-F" : f_ppl.item(),
            "PPL-R":  rt_ppl.item(),
            'model_id': cfg.model_id,
            'batch_size': cfg.batch_size * 4 * cfg.gradient_accumulation_steps ,
            'num_epochs': cfg.num_epochs,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
            'LoRA_r': cfg.LoRA_r,
            'LoRA_alpha': cfg.LoRA_alpha,
            }}

update_json_dict(f'./results/scores/{cfg.loss_type}_{cfg.exp_type}_results.json', results)