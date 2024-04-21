import copy
import torch
import numpy as np
from mmcv import Config
from torch.utils.data import ConcatDataset

from groma.data.datasets.coco import COCODet
from groma.data.datasets.refcoco_rec import RefCOCO
from groma.data.datasets.flickr import Flickr30k
from groma.data.datasets.grit import Grit
from groma.data.datasets.refcoco_cap import RefCOCOCap
from groma.data.datasets.llava import LLaVAInstruct
from groma.data.datasets.groma import GromaInstruct
from groma.data.datasets.visual_genome import SingleRoundVG, MultiRoundsVG
from groma.data.datasets.det_data import ClassAgnosticCoCo, ClassAgnosticSA1B


def build_multi_datasets(dataset_cfg_file, tokenizer=None, **kwargs):
    dataset_cfgs = Config.fromfile(dataset_cfg_file)
    dataset_cfgs = dataset_cfgs.datasets
    assert isinstance(dataset_cfgs, list)
    datasets = [build_dataset(cfg, tokenizer=tokenizer, **kwargs) for cfg in dataset_cfgs]
    return ConcatDataset(datasets)


def build_dataset(dataset_cfg, tokenizer=None, **kwargs):
    dataset_type = dataset_cfg.pop('type')
    ratio = dataset_cfg.pop('ratio', 1)
    conv_temp = dataset_cfg.pop('conv_temp', 'default')

    if dataset_type in ('coco_box', 'obj365_box', 'openimage_box', 'v3det_box'):
        dataset = ClassAgnosticCoCo(**dataset_cfg)
    elif dataset_type == 'sa1b_box':
        dataset = ClassAgnosticSA1B(**dataset_cfg)
    elif dataset_type == 'coco':
        dataset = COCODet(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'flickr30k':
        dataset = Flickr30k(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'single_vg':
        dataset = SingleRoundVG(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'multi_vg':
        dataset = MultiRoundsVG(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'refcoco_cap':
        dataset = RefCOCOCap(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'refcoco_rec':
        dataset = RefCOCO(**dataset_cfg, tokenizer=tokenizer, conv_temp=conv_temp)
    elif dataset_type == 'grit':
        dataset = Grit(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    elif dataset_type == 'llava_instruct':
        dataset = LLaVAInstruct(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    elif dataset_type == 'groma_instruct':
        dataset = GromaInstruct(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], conv_temp=conv_temp)
    else:
        raise NotImplementedError

    if ratio < 1:
        print(f'randomly sample {ratio} of the dataset {dataset_type}: {int(ratio * len(dataset))}')
        random_indices = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        return subsample_dataset

    return dataset


if __name__ == '__main__':
    # for quick test
    dataset_cfg_file = 'groma/data/configs/vl_finetune.py'
    train_datasets = build_multi_datasets(dataset_cfg_file, tokenizer=None, img_processor=None)
    print(len(train_datasets))
    train_datasets[0]
    import random
    for i in range(10):
        ind = random.randint(0, len(train_datasets))
        train_datasets[ind]

