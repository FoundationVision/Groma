# Copyright (c) OpenMMLab. All rights reserved.
import random
import torch
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh


def normalize_box_coordinates(bbox, img_shape):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    img_h, img_w = img_shape[:2]
    bbox_new = [(cx / img_w), (cy / img_h), (w / img_w), (h / img_h)]
    bbox_new = torch.clamp(torch.cat(bbox_new, dim=-1), min=0., max=1.)
    return bbox_new


class ClassAgnosticCoCo(CocoDataset):
    CLASSES = ('object',)
    PALETTE = None

    def __init__(
        self,
        ann_file=None,
        img_prefix=None,
        test_mode=False
    ):
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True
        )

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize',
                 img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomCrop',
                 crop_type='absolute_range',
                 crop_size=(448, 896),
                 allow_negative_crop=False),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False, override=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**dataset_cfg)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        gt_bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        gt_bboxes = normalize_box_coordinates(gt_bboxes, img_shape)
        data_dict = {
            'image': data_item['img'].data,
            'class_labels': data_item['gt_labels'].data,
            'bboxes': gt_bboxes,
            'ori_shape': data_item['img_metas'].data['ori_shape'],
            'source': 'coco'
        }
        return data_dict


class ClassAgnosticSA1B(CocoDataset):
    CLASSES = ('object',)
    PALETTE = None

    def __init__(
        self,
        ann_file=None,
        img_prefix=None,
        test_mode=False
    ):
        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True
        )

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize',
                 img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                 multiscale_mode='value',
                 keep_ratio=True),
            dict(type='RandomCrop',
                 crop_type='absolute_range',
                 crop_size=(448, 896),
                 allow_negative_crop=False),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False, override=True),
            dict(type='CustomFilterAnnotations', min_size=14 * 14, max_size=400 * 400),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**dataset_cfg)

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        gt_bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        gt_bboxes = normalize_box_coordinates(gt_bboxes, img_shape)
        data_dict = {
            'image': data_item['img'].data,
            'class_labels': data_item['gt_labels'].data,
            'bboxes': gt_bboxes,
            'ori_shape': data_item['img_metas'].data['ori_shape'],
            'source': 'sa1b'
        }
        return data_dict



