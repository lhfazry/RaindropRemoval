#!/bin/bash

CUDA_VISIBLE_DEVICES=$CUDA_ID python scripts/image_train.py \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --diffusion_steps 1000 \
    --dropout 0.1 \
    --image_size 64 \
    --learn_sigma True \
    --noise_schedule cosine \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --use_new_attention_order True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --batch_size $BATCH_SIZE \
    --num_samples 10 \
    --timestep_respacing 250 \
    --classifier_scale 1.0 \
    --classifier_path models/64x64_classifier.pt \
    --classifier_depth 4 \
    --model_path models/64x64_diffusion.pt 
    --data_dir datasets/LEVIR/A
