# Groma: Grounded Multimodal Assistant

<p align="left"> 
    <img src='docs/teaser.png' align="center" width="80%"> 
</p>

> [**Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models**](https://arxiv.org/abs/2404.13013)               
> Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, Xiaojuan Qi    
> *Project page ([https://groma-mllm.github.io](https://groma-mllm.github.io))*   


## Contents
- [Install](#installation)
- [Model](#model-weights)
- [Data](#prepare-data)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)


## Installation
Clone the repository
~~~
git clone https://github.com/FoundationVision/Groma.git
cd Groma
~~~

Create the conda environment and install dependencies
~~~
conda create -n groma python=3.9 -y
conda activate groma
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

cd mmcv
MMCV_WITH_OPS=1 pip install -e .
cd ..
~~~

Install falsh-attention for training
~~~
pip install ninja
pip install flash-attn --no-build-isolation
~~~


## Model Weights
To play with Groma, please download the [model weights](https://huggingface.co/FoundationVision/groma-7b-finetune) from huggingface. 

We additionally provide pretrained checkpoints from intermediate training stages. 
You can start from any point to customize training.

| Training stage | Required checkpoints |
|:--------------:|:--------------------:|
| Detection pretraining | [DINOv2-L](https://huggingface.co/facebook/dinov2-large) |
| Alignment pretraining | [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5), [Groma-det-pretrain](https://huggingface.co/FoundationVision/groma-det-pretrain) |
| Instruction finetuning | [Groma-7b-pretrain](https://huggingface.co/FoundationVision/groma-7b-pretrain) |



## Prepare Data
We provide instructions to download datasets used at different training stages of Groma, 
including [Groma Instruct](https://huggingface.co/datasets/FoundationVision/groma_instruct/),
a 30k viusally grounded conversation dataset constructed with GPT-4V.
You don't have to download all of them unless you want to train Groma from scratch.
Please follow instructions in [DATA.md](docs/DATA.md) to prepare datasets.

<table>
  <tr>
    <th align="left">Training stage</th>
    <th align="left">Data types</th>
    <th align="left">Datasets</th>
  </tr>
  <tr>
    <td align="left">Detection pretraining</td>
    <td align="left">Detection</td>
    <td align="left">COCO, Objects365, OpenImages, V3Det, SA1B</td>
  </tr>
  <tr>
    <td rowspan="4" align="left">Alignment pretraining</td>
    <td align="left">Image caption</td>
    <td align="left">ShareGPT-4V-PT</td>
  </tr>
  <tr>
    <td align="left">Grounded caption</td>
    <td align="left">Flickr30k Entities</td>
  </tr>
  <tr>
    <td align="left">Region caption</td>
    <td align="left">Visual Genome, RefCOCOg</td>
  </tr>
  <tr>
    <td align="left">REC</td>
    <td align="left">COCO, RefCOCO/g/+, Grit-20m</td>
  </tr>
  <tr>
    <td rowspan="4" align="left">Instruction finetuning</td>
    <td align="left">Grounded caption</td>
    <td align="left">Flickr30k Entities</td>
  </tr>
  <tr>
    <td align="left">Region caption</td>
    <td align="left">Visual Genome, RefCOCOg</td>
  </tr>
  <tr>
    <td align="left">REC</td>
    <td align="left">COCO, RefCOCO/g/+</td>
  </tr>
  <tr>
    <td align="left">Instruction following</td>
    <td align="left">Groma Instruct, LLaVA Instruct, ShareGPT-4V</td>
  </tr>
</table>


## Training
For detection pretraining, please run
~~~
bash scripts/det_pretrain.sh {path_to_dinov2_ckpt} {output_dir}
~~~

For alignment pretraining, please run
~~~
bash scripts/vl_pretrain.sh {path_to_vicuna_ckpt} {path_to_groma_det_pretrain_ckpt} {output_dir}
~~~

For instruction finetuing, please run
~~~
bash scripts/vl_finetune.sh {path_to_groma_7b_pretrain_ckpt} {output_dir}
~~~


## Inference
To test on single image, you can run
~~~
python -m llava.eval.run_groma \
    --model-name {path_to_groma_7b_finetune} \
    --image-file {path_to_img} \
    --query {user_query}
~~~


## Evaluation
For evaluation, please refer to [EVAL.md](docs/EVAL.md) for more details.


[comment]: <> (## Citation)

[comment]: <> (If you find this repo useful for your research, please consider citing our paper:)

[comment]: <> (```)

[comment]: <> (@inproceedings{ma2023codet,)

[comment]: <> (  title={CoDet: Co-Occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection},)

[comment]: <> (  author={Ma, Chuofan and Jiang, Yi and Wen, Xin and Yuan, Zehuan and Qi, Xiaojuan},)

[comment]: <> (  booktitle={Advances in Neural Information Processing Systems},)

[comment]: <> (  year={2023})

[comment]: <> (})

[comment]: <> (```)


## Acknowledgement
Groma is built upon the awesome works 
[LLaVA](https://github.com/haotian-liu/LLaVA/) and 
[GPT4ROI](https://github.com/jshilong/GPT4RoI).



## LICENSE
This project is licensed under the Apache License 2.0 - 
see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@misc{Groma,
      title={Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models}, 
      author={Chuofan Ma and Yi Jiang and Jiannan Wu and Zehuan Yuan and Xiaojuan Qi},
      year={2024},
      eprint={2404.13013},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
