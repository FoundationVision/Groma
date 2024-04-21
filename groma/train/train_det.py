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

import time
import mmcv
import torch
import pathlib
import transformers
from mmcv.runner import get_dist_info
from mmdet.apis.test import collect_results_cpu
from mmdet.core import bbox2result
from dataclasses import dataclass, field, astuple
from typing import Dict, Optional, Sequence, List
from transformers import Trainer, Dinov2Config, DeformableDetrConfig
from transformers.image_transforms import center_to_corners_format

from groma.model.ddetr import CustomDDETRConfig, CustomDDETRModel
from groma.data.build import build_multi_datasets
from groma.data.collator import DataCollatorForDetDataset
from groma.train.groma_trainer import GromaTrainer


@dataclass
class ModelArguments:
    vis_encoder: Optional[str] = field(default=None)
    zs_weight_path: Optional[str] = field(default=None)
    vis_output_layer: Optional[int] = field(default=-1)  # default to the last layer
    num_queries: Optional[int] = field(default=300)
    ddetr_hidden_dim: Optional[int] = field(default=256)
    num_encoder_layers: Optional[int] = field(default=6)
    num_decoder_layers: Optional[int] = field(default=6)
    num_feature_levels: Optional[int] = field(default=1)
    two_stage: Optional[bool] = field(default=True)
    with_box_refine: Optional[bool] = field(default=True)
    num_classes: Optional[int] = field(default=80)
    auxiliary_loss: Optional[bool] = field(default=True)
    match_class_cost: Optional[int] = field(default=2)
    match_bbox_cost: Optional[int] = field(default=5)
    match_giou_cost: Optional[int] = field(default=2)
    cls_loss_coefficient: Optional[int] = field(default=2)
    bbox_loss_coefficient: Optional[int] = field(default=5)
    giou_loss_coefficient: Optional[int] = field(default=2)
    focal_alpha: Optional[float] = field(default=0.25)


@dataclass
class DataArguments:
    dataset_config: str = field(default='groma/datasets/dataset_configs.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vis_encoder: Optional[bool] = field(default=True)
    freeze_ddetr: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    lr_backbone_names: Optional[tuple] = field(default=("vis_encoder",))
    lr_linear_proj_names: Optional[tuple] = field(default=('reference_points', 'sampling_offsets'))
    lr_multiplier: Optional[float] = field(default=0.1)
    group_by_data_source: Optional[bool] = field(default=True)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def post_process(outputs, target_sizes, threshold=0., top_k=100):
    out_logits, out_bbox = outputs.logits['coco'], outputs.pred_boxes
    if target_sizes is not None:
        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

    prob = out_logits.sigmoid()
    prob = prob.view(out_logits.shape[0], -1)
    k_value = min(top_k, prob.size(1))
    topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
    scores = topk_values
    topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
    labels = topk_indexes % out_logits.shape[2]
    boxes = center_to_corners_format(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    if isinstance(target_sizes, List):
        img_h = torch.Tensor([i[0] for i in target_sizes])
        img_w = torch.Tensor([i[1] for i in target_sizes])
    else:
        img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        score = s[s > threshold]
        label = l[s > threshold]
        box = b[s > threshold]
        results.append({"scores": score, "labels": label, "boxes": box})

    return results


def eval(model, test_dataloader, num_classes):
    model.eval()
    results = []
    dataset = test_dataloader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for data in test_dataloader:
        with torch.no_grad():
            target_size = data['ori_shapes'].to('cuda')
            result = model(images=data['images'].to('cuda'))
            result = post_process(result, target_size)
            result = [(torch.cat((x['boxes'], x['scores'].unsqueeze(1)), dim=1), x['labels']) for x in result]
            result = [bbox2result(det_bboxes, det_labels, num_classes) for det_bboxes, det_labels in result]

        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    results = collect_results_cpu(results, len(dataset), None)
    if rank == 0:
        dataset.evaluate(results)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    vis_encoder_cfg = Dinov2Config.from_pretrained(model_args.vis_encoder)
    ddetr_cfg = DeformableDetrConfig(
        d_model=model_args.ddetr_hidden_dim,
        encoder_layers=model_args.num_encoder_layers,
        decoder_layers=model_args.num_decoder_layers,
        num_feature_levels=model_args.num_feature_levels,
        two_stage=model_args.two_stage,
        two_stage_num_proposals=model_args.num_queries,
        num_queries=model_args.num_queries,
        num_labels=model_args.num_classes,
        auxiliary_loss=model_args.auxiliary_loss,
        with_box_refine=model_args.with_box_refine,
        class_cost=model_args.match_class_cost,
        bbox_cost=model_args.match_bbox_cost,
        giou_cost=model_args.match_giou_cost,
        cls_loss_coefficient=model_args.cls_loss_coefficient,
        bbox_loss_coefficient=model_args.bbox_loss_coefficient,
        giou_loss_coefficient=model_args.giou_loss_coefficient,
        focal_alpha=model_args.focal_alpha,
    )
    model_cfg = CustomDDETRConfig(
        vis_encoder_path=model_args.vis_encoder,
        zs_weight_path=model_args.zs_weight_path,
        vis_encoder_cfg=vis_encoder_cfg,
        ddetr_cfg=ddetr_cfg,
        vis_output_layer=model_args.vis_output_layer,
    )
    model = CustomDDETRModel(model_cfg)

    if training_args.freeze_vis_encoder:
        model.freeze_vis_encoder()
    if training_args.freeze_ddetr:
        model.freeze_ddetr()

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not match_name_keywords(n, training_args.lr_backbone_names) and
                       not match_name_keywords(n, training_args.lr_linear_proj_names) and p.requires_grad],
            "lr": training_args.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, training_args.lr_backbone_names) and p.requires_grad],
            "lr": training_args.learning_rate * training_args.lr_multiplier,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, training_args.lr_linear_proj_names) and p.requires_grad],
            "lr": training_args.learning_rate * training_args.lr_multiplier,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    train_dataset = build_multi_datasets(data_args.dataset_config)
    data_collator = DataCollatorForDetDataset()

    trainer = GromaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # from torch.utils.data import DataLoader
    # from torch.utils.data.distributed import DistributedSampler
    # from groma.data.datasets.det_data import ClassAgnosticCoCo
    # from groma.data.collator import DataCollatorForDetEvalDataset
    #
    # img_prefix = 'coco/val2017'
    # ann_file = 'class_agnostic_det/coco_instances_val2017.json'
    # test_dataset = ClassAgnosticCoCo(ann_file, img_prefix, test_mode=True)
    # data_collator_test = DataCollatorForDetEvalDataset()
    # sampler = DistributedSampler(test_dataset, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, collate_fn=data_collator_test, sampler=sampler, batch_size=8)
    #
    # eval(trainer.model, test_dataloader, model_args.num_classes)


if __name__ == "__main__":
    train()

