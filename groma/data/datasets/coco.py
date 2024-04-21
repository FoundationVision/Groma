import random
import torch
from collections import defaultdict
from mmdet.datasets import CocoDataset
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from groma.data.datasets.det_data import normalize_box_coordinates
from groma.constants import DEFAULT_TOKENS, IGNORE_INDEX
from groma.data.conversation import conv_templates


INSTRUCTIONS = [
    "Locate all {} in this image.",
    "Identify all instances of {} in the photo.",
    "Find all instances of {} in the image.",
    "Point out all the {} visible in this picture.",
    "Detect and list each {} that appears in this photo.",
    "What is the position of each {} in the image?"
]


class COCODet(CocoDataset):
    def __init__(
        self,
        ann_file=None,
        img_prefix=None,
        tokenizer=None,
        test_mode=False,
        conv_temp='default'
    ):
        self.tokenizer = tokenizer
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

        img_norm_cfg = dict(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            to_rgb=True
        )

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=448),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'img_info']),
        ]

        pipeline = test_pipeline if test_mode else train_pipeline
        dataset_cfg = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**dataset_cfg)

    def preprocess(self, data_item):
        image = data_item['img'].data
        labels = data_item['gt_labels'].data
        bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        bboxes = bbox_xyxy_to_cxcywh(bboxes)
        bboxes = normalize_box_coordinates(bboxes, img_shape)

        if len(labels) == 0:
            return None

        label2box = defaultdict(list)
        for i, label in enumerate(labels):
            label = self.CLASSES[label]
            label = label.strip().lower()
            label2box[label].append(bboxes[i])

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        label = random.choice(list(label2box.keys()))
        refexp = DEFAULT_TOKENS['boe'] + label + DEFAULT_TOKENS['eoe']
        instruct = random.choice(INSTRUCTIONS).format(refexp)
        answer = DEFAULT_TOKENS['sep']
        answer += DEFAULT_TOKENS['bor'] + DEFAULT_TOKENS['gbox'] * len(label2box[label]) + DEFAULT_TOKENS['eor']
        answer += DEFAULT_TOKENS['sep']
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))
        prompt = self.conv_temp.get_prompt(conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids[0]

        # Mask targets
        targets = input_ids.clone()
        sep_inds = (input_ids == self.seperator_id).nonzero(as_tuple=True)[0]
        assert len(sep_inds) % 2 == 0
        for i in range(0, len(sep_inds), 2):
            pre_sep = 0 if i == 0 else sep_inds[i - 1]
            cur_sep = sep_inds[i]
            targets[pre_sep:cur_sep] = IGNORE_INDEX
        eos_inds = (input_ids == self.eos_id).nonzero(as_tuple=True)[0]
        targets[eos_inds[1:]] = self.eos_id

        # Remove sep token
        mask = input_ids != self.seperator_id
        input_ids = input_ids[mask]
        targets = targets[mask]

        data_dict = dict(
            input_ids=input_ids,
            labels=targets,
            image=image,
            source='coco',
            ground_boxes=torch.stack(label2box[label]),
            img_metas=data_item['img_metas'].data
        )
        return data_dict

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        data_dict = self.preprocess(data_item)
        if data_dict is None:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)
        return data_dict
