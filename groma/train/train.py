# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import LlamaConfig, AutoImageProcessor

from groma.model.ddetr import CustomDDETRConfig
from groma.model.groma import GromaConfig, GromaModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForHybridDataset
from groma.constants import DEFAULT_TOKENS, REGION_IDX_TOKENS
from groma.train.groma_trainer import GromaTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)
    perceiver: Optional[str] = field(default=None)
    nms_thres: Optional[float] = field(default=0.6)
    box_score_thres: Optional[float] = field(default=0.)
    max_region_num: Optional[int] = field(default=100)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/datasets/dataset_configs.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_llm: bool = field(default=False)
    freeze_perceiver: bool = field(default=True)
    freeze_vl_bridge: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=512)
    use_custom_lr: bool = field(default=False)
    custom_lr_params: tuple = field(default=('perceiver', 'llm'))
    custom_lr: float = field(default=2e-5)
    group_by_data_source: Optional[bool] = field(default=True)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.model_name_or_path:
        # to check loader tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False
        )
        model = GromaModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        vis_processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path)
        model.init_special_token_id(tokenizer)
    elif model_args.llm and model_args.perceiver:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.llm,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False
        )
        num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + REGION_IDX_TOKENS, special_tokens=True)
        tokenizer.pad_token = DEFAULT_TOKENS['pad']
        vis_processor = AutoImageProcessor.from_pretrained(model_args.perceiver)
        model_cfg = GromaConfig(
            llm_cfg=LlamaConfig.from_pretrained(model_args.llm),
            perceiver_cfg=CustomDDETRConfig.from_pretrained(model_args.perceiver),
            num_new_token=num_new_token,
            nms_thres=model_args.nms_thres,
            box_score_thres=model_args.box_score_thres,
            max_region_num=model_args.max_region_num
        )
        model = GromaModel(
            model_cfg,
            pretrained_perceiver=model_args.perceiver,
            pretrained_llm=model_args.llm,
        )
        model.init_special_token_id(tokenizer)
        model.llm.config.use_cache = False
        model.generation_config = model.llm.generation_config
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
    else:
        raise ValueError("Should specify either the pretrained model or the model config.")

    if training_args.freeze_perceiver:
        model.freeze_perceiver()
    if training_args.freeze_vl_bridge:
        model.freeze_vl_bridge()
    if training_args.freeze_llm:
        model.freeze_llm()

    train_datasets = build_multi_datasets(
        data_args.dataset_config,
        tokenizer=tokenizer,
        img_processor=vis_processor)
    data_collator = DataCollatorForHybridDataset(tokenizer)

    trainer = GromaTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=data_collator
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model()
    trainer.save_state()
    vis_processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
