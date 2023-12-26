# Making Large Language Models Perform Better in Knowledge Graph Completion
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/KoPA/main/LICENSE)
[![AAAI](https://img.shields.io/badge/Preprint'23-brightgreen)](https://arxiv.org/abs/2310.06671)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
 - [Making Large Language Models Perform Better in Knowledge Graph Completion
](https://arxiv.org/abs/2310.06671)

> Large language model (LLM) based knowledge graph completion (KGC) aims to predict the missing triples in the KGs with LLMs and enrich the KGs to become better web infrastructure, which can benefit a lot of web-based automatic services. However, research about LLM-based KGC is limited and lacks effective utilization of LLM's inference capabilities, which ignores the important structural information in KGs and prevents LLMs from acquiring accurate factual knowledge. In this paper, we discuss how to incorporate the helpful KG structural information into the LLMs, aiming to achieve structrual-aware reasoning in the LLMs. We first transfer the existing LLM paradigms to structural-aware settings and further propose a knowledge prefix adapter (KoPA) to fulfill this stated goal. KoPA employs structural embedding pre-training to capture the structural information of entities and relations in the KG. Then KoPA informs the LLMs of the knowledge prefix adapter which projects the structural embeddings into the textual space and obtains virtual knowledge tokens as a prefix of the input prompt. We conduct comprehensive experiments on these structural-aware LLM-based KGC methods and provide an in-depth analysis comparing how the introduction of structural information would be better for LLM's knowledge reasoning ability.

## 🌈 Model Architecture
![Model_architecture](figure/model.png)


## 🔬 Dependencies
Our code is developed based on [alpaca-lora](https://github.com/tloen/alpaca-lora). Please build the Python following the instruction in Alpaca-lora.

Some core python library config: 
- Python 3.9.16
- torch 2.0.0
- transformers 4.28.0

## 🌲 Data Preparation
Due to the size of the data, you need to download and unzip the data file data.zip from [this link](https://drive.google.com/file/d/1J1Ioi23jTMaBkBDYzfIy2MAZYMUIjFWW/view?usp=drive_link) and put them in the data/.


## 📕 Training & Test
- Note: The current dataset is just a little demonstration to help you run the full pipeline. We will release the full datasets in the future.

- run KoPA tuning
```shell
export WANDB_DISABLED=true
wandb offline
# For CoDeX-S dataset
CUDA_VISIBLE_DEVICES=0 nohup python finetune_kopa.py \
    --base_model 'YOUR LLM PATH' \
    --data_path 'data/CoDeX-S-train.json' \
    --output_dir 'YOUR SAVE PATH' \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 12 \
    --micro_batch_size 12 \
    --num_prefix 1 \
    --kge_model 'data/CoDeX-S-rotate.pth' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log.txt &


# For CoDeX-S dataset
export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=0 nohup python finetune_kopa.py \
    --base_model 'YOUR LLM PATH' \
    --data_path 'data/CoDeX-S-train.json' \
    --output_dir 'YOUR SAVE PATH' \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 12 \
    --micro_batch_size 12 \
    --num_prefix 1 \
    --kge_model 'data/CoDeX-S-rotate.pth' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log.txt &
```
You may need to fill the LLM path and save path before running. The hyper-parameters can be tuned by yourself.

- run inference
```shell
CUDA_VISIBLE_DEVICES=0 python inference_kopa.py
```


## 🤝 Cite:
Please condiser citing this paper if you use the code from our work.
Thanks a lot :)

```bigquery
@misc{zhang2023making,
      title={Making Large Language Models Perform Better in Knowledge Graph Completion}, 
      author={Yichi Zhang and Zhuo Chen and Wen Zhang and Huajun Chen},
      year={2023},
      eprint={2310.06671},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
