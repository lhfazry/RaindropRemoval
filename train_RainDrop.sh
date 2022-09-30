#!/bin/bash
export OPENAI_LOGDIR=logs/RainDrop

echo "CUDA_ID: " $CUDA_ID
echo "BATCH_SIZE: " $BATCH_SIZE

CUDA_VISIBLE_DEVICES=$CUDA_ID python scripts/image_train.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 1000 \
    --image_size 128 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 256 \
    --num_heads 4 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --batch_size $BATCH_SIZE \
    --data_dir datasets/RainDrop/train/gt