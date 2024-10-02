#!/bin/bash
devices=0,1,2,3

# Change the ckpt_path to stage 1 output and specify output dir
ckpt_path=exp_dirs/example_exp/hmdb8_umt_stage1/checkpoint-latest.pth
output_dir=exp_dirs/example_exp/arid-hmdb_source_ft_stage2

epochs=50
warmup_epochs=$((epochs / 5))  # Calculate warmup_epochs as 1/5th of epochs

export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + $RANDOM % 20000))
CUDA_VISIBLE_DEVICES=${devices}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "Using $num_gpus GPUs"
export OMP_NUM_THREADS=1

torchrun \
--nproc_per_node=$num_gpus --nnodes=1 --master_port=${MASTER_PORT} \
run_stage2.py \
    --config configs/stage2_config.yaml \
    --output_dir "${output_dir}" \
    --dataset arid-hmdb \
    --finetune ${ckpt_path} \
    --frozen_layers '' \
    --freeze_patch_embedding false \
    --warmup_epochs ${warmup_epochs} \
    --epochs ${epochs} \
    --batch_size 7 \
    --disable_wandb \
    --wandb_entity your_wandb_entity \
    --wandb_project your_wandb_project \
    --wandb_group your_wandb_group \
    --no_auto_reload \
    --eval_freq 5 \
    --save_ckpt \
    --seed 0
