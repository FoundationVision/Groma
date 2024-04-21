datasets = [
    {
        'type': 'refcoco_rec',
        'ann_file': 'refcoco_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'llava'
    },
    {
        'type': 'refcoco_rec',
        'ann_file': 'refcoco+_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'llava'
    },
    {
        'type': "refcoco_rec",
        'ann_file': 'refcocog_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'llava'
    },
    {
        'type': "flickr30k",
        'ann_file': 'flickr_entities_train.json',
        'img_prefix': 'dataset/flickr30k/images/',
        'conv_temp': 'llava'
    },
    {
        'type': "llava_instruct",
        'ann_file': 'llava_conversation_reasoning.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'llava'
    },
    {
        'type': "llava_instruct",
        'ann_file': 'sharegpt4v_instruct_gpt4-vision_cap100k_new.json',
        'img_prefix': 'dataset/sharegpt4v/data',
        'ratio': 0.23,
        'conv_temp': 'llava'
    },
    {
        'type': "multi_vg",
        'ann_file': 'vg_train_multi.json',
        'img_prefix': 'dataset/visual_genome/images/',
        'conv_temp': 'llava'
    },
    {
        'type': "refcoco_cap",
        'ann_file': 'refcocog_train.json',
        'img_prefix': 'dataset/coco/train2017',
        'conv_temp': 'llava'
    },
    {
        'type': "coco",
        'ann_file': 'instances_train2017.json',
        'img_prefix': 'dataset/coco/train2017',
        'ratio': 0.5,
        'conv_temp': 'llava'
    },
    {
        'type': "groma_instruct",
        'ann_file': 'groma_instruct.json',
        'img_prefix': 'dataset/visual_genome/images/',
        'conv_temp': 'llava'
    }
]