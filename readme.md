# EFFECTIVE UNLEARNING IN LLMS RELIES ON THE RIGHT DATA RETENTION STRATEGY

### Abstract
Unlearning in Large Language Models (LLMs) has gained increasing attention in recent years due to its critical role in ensuring ethical and legal compliance. Although significant progress has been made in developing unlearning algorithms, relatively little attention has been devoted to the data perspective. In particular, the role of retain-set selection in preserving model utility remains underexplored, even though it is critical for making unlearning practical in real-world applications. In this work, we explore strategies for constructing effective retain sets by adapting methods from coreset selection and prior unlearning research. We evaluate these approaches on two complementary datasets: (i) a monotonic dataset built from a benchmark dataset, and (ii) a mixed, larger-scale dataset combining WPU, TOFU, and Dolly, which better reflects realistic scenarios where forget and retain samples are not explicitly defined. We find that both model utility and forget quality are strongly influenced by the variance of the model’s representations within the selected retain set. Moreover, we show that simply choosing data samples with high semantic or syntactic similarity to the forget set can yield substantially better results than standard coreset techniques. To the best of our knowledge, this work represents the first systematic study of retain-set selection for LLM unlearning, highlighting both its importance and the challenges it poses in practical settings.


### Create Environment
```bash
conda create -n corest python=3.11
conda activate coreset
pip install -r requirements.txt
```



### Finetune

To reproduce the results, the first step is to finetune the `Llama3.1-8B Instruct` model. We fine-tuned the model for 10 epochs with maximum learning rate of `2e-5` and batch size of 32. We used the original `meta-llama/Llama-3.1-8B-Instruct` HF repo. If you use the same hf repo, please update **access token** in the `Config_ft` class from ```config.py``` file. We need to finetune on two datasets giving us two different models (WPU, Mix). Based on this please select the right data path

```bash
Step 1: Change the max_length based on the dataset (WPU = 256, Mix = 512)
Step 2: Change the dataset path correctly.
python finetune.py
```

```bash
Step 1: Update the loss_type and exp_type (both pre_unlearning) in the Config/Config2 classes in config.py file
Step 2: Update line 23 in eval.py file based on the dataset (cfg = Config() for WPU, cfg = Config2() for Mix).
python eval.py
```


### Unlearn

There are overall 40 experiments just with Gradient difference. The datasets used for the experiments can be found in the `/data` folder. There are two Config classes in the ```config.py```. `Config` for **WPU** and `Config2` for **Mix** datasets. Based on your experiments, please select the right Config Class. 


```bash
Step 1: Update the self.loss_type and exp_type in Config Class of config.py to gd
Step 2: Select the retain dataset path (grand, moderate, etc).
Step 3: Select the appropriate epochs from the paper (In the Appendix, Table:4) for the experiment.
Step 4: Please check if the model_id path is correct (finetuned version WPU/Mix).
Step 5: Change the config line code in the gd.py file (line 21, cfg = Config() for WPU, cfg = Config2() for Mix).
```


For gradient difference 

```bash
python gd.py
```
Without changing anything in the config, please follow the next steps for the evaluation. 

```bash
Step 1: Update line 23 in eval.py file based on the dataset (cfg = Config() for WPU, cfg = Config2() for Mix).
python eval.py
```