#!/bin/bash
devices=0,1,2,3

# Change the ckpt_path to stage 2 output and specify output dir
ckpt_path=exp_dirs/example_exp/arid-hmdb_source_ft_stage2/checkpoint-latest.pth
output_dir=exp_dirs/example_Exp/arid-hmdb_cst_stage3

thresh=0.1
epochs=20
warmup_epochs=$((epochs / 5))  # Calculate warmup_epochs as 1/5th of epochs

export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + $RANDOM % 20000))
CUDA_VISIBLE_DEVICES=${devices}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "Using $num_gpus GPUs"
export OMP_NUM_THREADS=1

torchrun \
--nproc_per_node=$num_gpus --nnodes=1 --master_port=${MASTER_PORT} \
run_stage3.py \
    --config configs/stage3_config.yaml \
    --output_dir "${output_dir}" \
    --dataset arid-hmdb \
    --student_init ${ckpt_path} \
    --clip_threshold ${thresh} \
    --epochs ${epochs} \
    --warmup_epochs ${warmup_epochs} \
    --batch_size 5 \
    --val_interval 1 \
    --save_ckpt_freq 10 \
    --no_auto_resume \
    --num_workers 6 \
    --initial_validation \
    --disable_wandb \
    --wandb_entity your_wandb_entity \
    --wandb_project your_wandb_project \
    --wandb_group your_wandb_group \
    --seed 0