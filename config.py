# for wpu
class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.loss_type      = 'gd' # change this with the experiment types provided above
        self.access_token   = '' # huggingface token
        self.model_id       = './outputs/wpu_llama' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 8 #please adjust this along with gradient_accumulation_steps for batch size
        self.gradient_accumulation_steps = 1 
        self.num_epochs     = 25
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 256
        self.exp_type       = 'semantic_5' #change this with the experiment types provided in the paper
        self.save_dir       = f'outputs/{self.loss_type}_{self.exp_type}_model' 
        self.retriever_model= 'thenlper/gte-small'
        self.forget_path    = './data/wpu_data/forget_1.csv' 
        self.retain_path    = './data/wpu_data/semantic/semantic_5.csv' #change this path
        self.retain_full    = './data/wpu_data/retain_100.csv'
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0
        


# for mix
class Config2:
    def __init__(self):
        super(Config2, self).__init__()
        self.loss_type      = 'gd' # change this with the experiment types provided above
        self.access_token   = '' 
        self.model_id       = './outputs/mix_llama' 
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 1e-5
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 8 #please adjust this along with gradient_accumulation_steps for batch size
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 512
        self.exp_type       = 'semantic_5' #change this with the experiment types provided in the paper
        self.save_dir       = f'outputs/unified/{self.loss_type}_{self.exp_type}_model' 
        self.retriever_model= 'thenlper/gte-small'
        self.forget_path    = './data/mix/forget_1.csv' 
        self.retain_path    = './data/mix/semantic/semantic_5.csv'
        self.retain_full    = './data/mix/test.csv'
        self.npo_beta       = 0.1
        self.npo_retain_alpha = 1.0
        self.npo_forget_gamma = 1.0





## this is for finetuning
class Config_ft:
    def __init__(self):
        super(Config_ft, self).__init__()
        self.model_id       = 'meta-llama/Llama-3.1-8B-Instruct'
        self.access_token   = '' # huggingface token 
        self.LoRA_r         = 64 
        self.LoRA_alpha     = 64 
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05 
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', ' q_proj', 'down_proj']
        self.batch_size     = 32 #please adjust this along with gradient_accumulation_steps for batch size
        self.gradient_accumulation_steps = 1 
        self.num_epochs     = 10
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.exp_type       = 'ckpt_desc'
        self.model_name    = 'llama_8b'
        self.save_dir       = 'outputs/wpu_llama' #'outputs/mix_llama'
        self.max_length     = 256 #change this to 512 when fine-tuning mixed data
        self.data_path      = './data/wpu_data/full_dataset_100.csv' #'./data/mix/full_data.csv'
