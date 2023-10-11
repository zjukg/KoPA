export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=1 nohup python finetune.py \
    --base_model 'YOUR LLM PATH' \
    --data_path 'preprocess/UMLS-train.json' \
    --output_dir 'YOUR SAVE PATH' \
    --num_epochs 3 \
    --batch_size 12 \
    --micro_batch_size 12 \
    --lora_r 32 \
    --learning_rate 3e-4 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log.txt &

