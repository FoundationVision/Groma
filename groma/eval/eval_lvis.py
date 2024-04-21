import os
import json
import torch
import random
import argparse
import torchvision
from tqdm import tqdm
from lvis import LVISEval
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer

from groma.utils import disable_torch_init
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.datasets.lvis import LVISDet
from groma.data.datasets.coco import INSTRUCTIONS


class CustomLVISEval(LVISEval):
    def summarize(self):
        """Compute and display summary metrics for evaluation results."""
        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results["AP"] = self._summarize('ap')
        self.results["AP50"] = self._summarize('ap', iou_thr=0.50)
        self.results["AP75"] = self._summarize('ap', iou_thr=0.75)
        self.results["APs"] = self._summarize('ap', area_rng="small")
        self.results["APm"] = self._summarize('ap', area_rng="medium")
        self.results["APl"] = self._summarize('ap', area_rng="large")
        self.results["APr"] = self._summarize('ap', freq_group_idx=0)
        self.results["APc"] = self._summarize('ap', freq_group_idx=1)
        self.results["APf"] = self._summarize('ap', freq_group_idx=2)

        key = "AR@{}".format(max_dets)
        self.results[key] = self._summarize('ar')
        self.results["AR50"] = self._summarize('ar', iou_thr=0.50)
        self.results["AR75"] = self._summarize('ar', iou_thr=0.75)

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results[key] = self._summarize('ar', area_rng=area_rng)


class LVISTest(LVISDet):
    def _parse_ann_info(self, img_info, ann_info):
        ann = super()._parse_ann_info(img_info, ann_info)
        ann['id'] = img_info['id']
        return ann

    def preprocess(self, data_item):
        image = data_item['img'].data
        label = data_item['gt_labels'].data.tolist()[0]
        img_id = data_item['img_info']['id']
        img_shape = data_item['img_metas'].data['ori_shape'][:2]
        cat_name = self.CLASSES[label].replace('_', ' ').strip().lower()

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        refexp = DEFAULT_TOKENS['boe'] + cat_name + DEFAULT_TOKENS['eoe']
        instruct = random.choice(INSTRUCTIONS).format(refexp, refexp)
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], ''))
        prompt = self.conv_temp.get_prompt(conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids

        data_dict = dict(
            input_ids=input_ids,
            image=image,
            label=label,
            img_id=img_id,
            img_shape=img_shape)
        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    image = batch[0]['image'].unsqueeze(dim=0)
    label = batch[0]['label']
    img_id = batch[0]['img_id']
    img_shape = batch[0]['img_shape']
    return input_ids, image, label, img_id, img_shape


def rescale_box(boxes, img_shape):
    h, w = img_shape
    boxes[:, 0] = boxes[:, 0] * w
    boxes[:, 1] = boxes[:, 1] * h
    boxes[:, 2] = boxes[:, 2] * w
    boxes[:, 3] = boxes[:, 3] * h
    return boxes


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres

    dataset = LVISTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='groma'
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
        sampler=sampler, collate_fn=custom_collate_fn)
    label2cat = {v: k for k, v in dataset.cat2label.items()}

    results = []
    invalid = 0
    for input_ids, image, label, img_id, img_shape in tqdm(dataloader):
        input_ids = input_ids.cuda()
        image = image.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=10,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )
        output_ids = outputs.sequences
        pred_boxes = outputs.hidden_states[0][-1]['pred_boxes'][0].cpu()
        input_token_len = input_ids.shape[1]
        predicted_box_tokens = [id for id in output_ids[0, input_token_len:] if id in model.box_idx_token_ids]
        selected_box_inds = [model.box_idx_token_ids.index(id) for id in predicted_box_tokens]
        selected_box_inds = [id for id in selected_box_inds if id < len(pred_boxes)]
        if len(selected_box_inds) == 0:
            invalid += 1
            continue
        selected_boxes = pred_boxes[selected_box_inds]
        selected_boxes = torchvision.ops.box_convert(selected_boxes, 'cxcywh', 'xywh')
        selected_boxes = rescale_box(selected_boxes, img_shape).tolist()
        for box in selected_boxes:
            result = {
                "image_id": img_id,
                "category_id": label2cat[label],
                "bbox": box,
                "score": 1.0
            }
            results.append(result)

    lvis_eval = CustomLVISEval(args.ann_file, args.result_file, 'bbox')
    lvis_eval.run()
    lvis_eval.print_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--ann-file", type=str, default="lvis_ground.json")
    parser.add_argument("--img-prefix", type=str, default="datasets/coco/")
    parser.add_argument("--box_score_thres", type=float, default=0.15)
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    args = parser.parse_args()

    eval_model(args)
