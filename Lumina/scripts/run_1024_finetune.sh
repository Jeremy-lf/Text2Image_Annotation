#!/usr/bin/env sh

train_data_path='./configs/data.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=/root/paddlejob/workspace/env_run/output/lvfeng/model/models--Alpha-VLLM--Lumina-Accessory/snapshots/711d5d6656c62957e8625b02ea53cc74f2c5589d/consolidated.00-of-01.pth
global_batch_size=8
micro_batch_size=1
snr_type=lognorm
lr=1
wd=0.01
precision=bf16
# training_type=full_model
training_type=lora

dir_name=lumina_results
mkdir -p "$dir_name"

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=25901 finetune_accessory_lf.py \
    --global_bsz_1024 ${global_batch_size} \
    --micro_bsz_1024 ${micro_batch_size} \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 --wd ${wd} \
    --data_path ${train_data_path} \
    --results_dir "$dir_name" \
    --data_parallel sdp \
    --max_steps 50000 \
    --ckpt_every 4000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20230122 \
    --num_workers 12 \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    --training_type ${training_type} \
    --cache_data_on_disk
