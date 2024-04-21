# Groma Datasets

To set up for training, please download the datasets and update the paths in dataset configs,
e.g., `groma/data/configs/vl_finetune.py`.
<table>
    <tr>
        <th>Dataset</th>
        <th>Images</th>
        <th>Annotations</th>
    </tr>
    <tr>
        <th colspan="3">VL Pretrain & Finetune</th>
    </tr>
    <tr>
        <th>COCO</th>
        <th rowspan="5"><a href="https://cocodataset.org/#download">coco_train_2017</a></th>
        <th><a href="https://cocodataset.org/#download">instances_train2017.json</a></th>
    </tr>
    <tr>
        <th>RefCOCO</th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">refcoco_train.json</a></th>
    </tr>
    <tr>
        <th>RefCOCO+</th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">refcoco+_train.json</a></th>
    </tr>
    <tr>
        <th>RefCOCOg</th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">refcocog_train.json</a></th>
    </tr>
    <tr>
        <th>LLaVA Instruct</th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">llava_conversation_reasoning.json</a></th>
    </tr>
    <tr>
        <th>Flickr30k Entities</th>
        <th><a href="https://shannon.cs.illinois.edu/DenotationGraph/">flickr_images</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">flickr_entities_train.json</a></th>
    </tr>
    <tr>
        <th>Visual Genome*</th>
        <th rowspan="2"><a href="https://homes.cs.washington.edu/~ranjay/visualgenome/api.html">vg_part1&2</a></th>
        <th>
            <a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">vg_train_single.json</a>,
            <a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">vg_train_multi.json</a>
        </th>
    </tr>
    <tr>
        <th>Groma Instruct</th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_instruct/tree/main">groma_instruct.json</a></th>
    </tr>
    <tr>
        <th>ShareGPT4V-PT</th>
        <th rowspan="2"><a href="https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md">sharegpt4v_data</a></th>
        <th><a href="https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main">share-captioner_coco_lcs_sam_1246k_1107.json</a></th>
    </tr>
    <tr>
        <th>ShareGPT4V</th>
        <th><a href="https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main">sharegpt4v_instruct_gpt4-vision_cap100k.json</a></th>
    </tr>
    <tr>
        <th>GRIT-20m</th>
        <th><a href="https://huggingface.co/datasets/zzliang/GRIT">grit_images</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">grit_filtered_10m.json</a></th>
    </tr>
    <tr>
        <th colspan="3">Detection Pretrain</th>
    </tr>
    <tr>
        <th>COCO</th>
        <th><a href="https://cocodataset.org/#download">coco_train_2017</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">class_agnostic_coco_instances_train2017.json</a></th>
    </tr>
    <tr>
        <th>Objects365</th>
        <th><a href="https://www.objects365.org/overview.html">objects365_v2</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">class_agnostic_obj365v2_train_new.json</a></th>
    </tr>
    <tr>
        <th>OpenImages</th>
        <th><a href="https://storage.googleapis.com/openimages/web/download_v6.html">openimages_v6</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">class_agnostic_openimages_v6_train_bbox.json</a></th>
    </tr>
    <tr>
        <th>V3Det</th>
        <th><a href="https://v3det.openxlab.org.cn/">v3det_v1</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">class_agnostic_v3det_2023_v1_train.json</a></th>
    </tr>
    <tr>
        <th>SA1B</th>
        <th><a href="https://ai.meta.com/datasets/segment-anything-downloads/">sa1b_images</a></th>
        <th><a href="https://huggingface.co/datasets/FoundationVision/groma_data/tree/main">class_agnostic_sa1b_2m.json</a></th>
    </tr>
</table>

*Note: Please put part_1 and part_2 images under the same folder.

