#!/bin/bash

DINO_PATH=$1
OUTPUT_DIR=$2
mkdir -p $OUTPUT_DIR
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    groma/train/train_det.py \
    --vis_encoder $DINO_PATH \
    --dataset_config groma/data/configs/det_pretrain.py \
    --bf16 True \
    --tf32 True \
    --num_classes 1 \
    --num_queries 300 \
    --two_stage True \
    --with_box_refine True \
    --ddetr_hidden_dim 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_feature_levels 1 \
    --freeze_vis_encoder True \
    --num_train_epochs 12 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --warmup_steps 100 \
    --logging_steps 100 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --dataloader_num_workers 8 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --report_to none \
    --output_dir $OUTPUT_DIR \
    | tee $OUTPUT_DIR/train.log
