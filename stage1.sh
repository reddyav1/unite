devices=0,1,2,3

# Change the output_dir and init_ckpt_dir
output_dir=exp_dirs/example_exp/hmdb8_umt_stage1/
init_ckpt_dir=/path/to/checkpoints/

export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + $RANDOM % 20000))
CUDA_VISIBLE_DEVICES=${devices}
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "Using $num_gpus GPUs"
export OMP_NUM_THREADS=1

torchrun \
--nproc_per_node=$num_gpus --nnodes=1 --master_port=${MASTER_PORT} \
run_stage1.py \
    --config configs/stage1_config.yaml \
    --output_dir "${output_dir}" \
    --dataset hmdb_sourceonly \
    --clip_loss_data source \
    --clip_decoder_init "${init_ckpt_dir}b16_ptk710_f8_res224.pth" \
    --student_init "${init_ckpt_dir}b16_ptk710_f8_res224.pth" \
    --epochs 100 \
    --warmup_epochs 10 \
    --batch_size 64 \
    --checkpoints_enabled \
    --save_ckpt_freq 50 \
    --num_workers 10 \
    --disable_wandb \
    --wandb_entity your_wandb_entity \
    --wandb_project your_wandb_project \
    --wandb_group your_wandb_group \
    --seed 0