datasets = [
    {
        'type': 'llava_instruct',
        'ann_file': 'share-captioner_coco_lcs_sam_1246k_new.json',
        'img_prefix': 'dataset/sharegpt4v/data',
        'conv_temp': 'default'
    },
    {
        'type': 'refcoco_rec',
        'ann_file': 'refcoco_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'default'
    },
    {
        'type': 'refcoco_rec',
        'ann_file': 'refcoco+_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'default'
    },
    {
        'type': "refcoco_rec",
        'ann_file': 'refcocog_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'default'
    },
    {
        'type': "flickr30k",
        'ann_file': 'flickr_entities_train.json',
        'img_prefix': 'dataset/flickr30k/images/',
        'conv_temp': 'default'
    },
    {
        'type': "single_vg",
        'ann_file': 'vg_train_single.json',
        'img_prefix': 'dataset/visual_genome/images/',
        'conv_temp': 'default',
        'ratio': 0.2
    },
    {
        'type': "grit",
        'ann_file': 'grit_filtered_10m.json',
        'img_prefix': 'dataset/grit-20m/images/',
        'conv_temp': 'default',
        'ratio': 0.1
    },
    {
        'type': "refcoco_cap",
        'ann_file': 'refcocog_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'default',
    },
    {
        'type': "coco",
        'ann_file': 'instances_train2017.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'default',
    },
]
