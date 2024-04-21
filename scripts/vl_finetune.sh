#!/bin/bash

PRETRAIN_PATH=$1
OUTPUT_DIR=$2
mkdir -p $OUTPUT_DIR
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    groma/train/train_mem.py \
    --model_name_or_path $PRETRAIN_PATH \
    --dataset_config groma/data/configs/vl_finetune.py \
    --freeze_perceiver True \
    --freeze_llm False \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config scripts/fsdp_config.json \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --model_max_length 2048 \
    --report_to none \
    --dataloader_num_workers 8 \
    --box_score_thres 0.15 \
    | tee $OUTPUT_DIR/train.log