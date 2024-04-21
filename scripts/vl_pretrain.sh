#!/bin/bash

LLM_PATH=$1
PERCEIVER_PATH=$2
OUTPUT_DIR=$3
mkdir -p $OUTPUT_DIR
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    groma/train/train_mem.py \
    --llm $LLM_PATH \
    --perceiver $PERCEIVER_PATH \
    --dataset_config groma/data/configs/vl_pretrain.py \
    --freeze_perceiver True \
    --freeze_llm True \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --report_to none \
    --dataloader_num_workers 8 \
    --box_score_thres 0.15 \
    | tee $OUTPUT_DIR/train.log




