# Evaluation


## REC Benchamrks
For referring expression evaluation on RefCOCO/g/+, 
please download [test annotations](https://huggingface.co/datasets/FoundationVision/groma_data/tree/main)
(e.g., `refcoco_val.json`) and run
~~~
torchrun --nnodes=1 --nproc_per_node={num_gpus} groma/eval/eval_rec.py \
    --model-name {path_to_groma_ckpts} \
    --img-prefix {path_to_coco_train_2017} \
    --ann-file {path_to_rec_test_annotation}
~~~

## LVIS-Ground Benchmark
To evaluate on LVIS-Ground proposed in the paper, 
please download [lvis_test.json](https://huggingface.co/datasets/FoundationVision/groma_data/tree/main)
and run
~~~
python groma/eval/eval_lvis.py \
    --model-name {path_to_groma_ckpts} \
    --img-prefix {path_to_coco_train_2017} \
    --ann-file {path_to_lvis_test_annotation}
~~~

## Referring Benchmarks
For region captioning evalution on Visual Genome, please download
[vg_test.json](https://huggingface.co/datasets/FoundationVision/groma_data/tree/main)
and run
~~~
torchrun --nnodes=1 --nproc_per_node={num_gpus} groma/eval/model_vg.py \
    --model-name {path_to_groma_ckpts} \
    --img-prefix {path_to_vg_images} \
    --ann-file {path_to_vg_test_annotation} \
    --result-dir {path_to_intermediate_result_dir}
  
python groma/eval/eval_cap.py \
    --ann-file {path_to_vg_test_annotation} \
    --result-dir {path_to_intermediate_result_dir}
~~~

Similarly, for RefCOCOg, download [refcocog_cap_val.json](https://huggingface.co/datasets/FoundationVision/groma_data/tree/main) and run
~~~
torchrun --nnodes=1 --nproc_per_node={num_gpus} groma/eval/model_refcocog.py \
    --model-name {path_to_groma_ckpts} \
    --img-prefix {path_to_coco_train_2017} \
    --ann-file {path_to_refcocog_test_annotation} \
    --result-dir {path_to_intermediate_result_dir}
  
python groma/eval/eval_cap.py \
    --ann-file {path_to_refcocog_test_annotation} \
    --result-dir {path_to_intermediate_result_dir}
~~~