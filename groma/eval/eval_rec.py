import os
import torch
import random
import argparse
from torchvision.ops import box_iou
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from transformers.image_transforms import center_to_corners_format
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from groma.utils import init_distributed_mode
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.datasets.refcoco_rec import RefCOCO, INSTRUCTIONS
from groma.data.datasets.det_data import normalize_box_coordinates


class RefCOCOTest(RefCOCO):
    def preprocess(self, data_item):
        image = data_item['img'].data
        label = data_item['gt_labels'][0]
        bboxes = data_item['gt_bboxes'].data
        img_shape = data_item['img_metas'].data['img_shape']
        bboxes = bbox_xyxy_to_cxcywh(bboxes)
        bboxes = normalize_box_coordinates(bboxes, img_shape)

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        refexp = DEFAULT_TOKENS['boe'] + label.strip() + DEFAULT_TOKENS['eoe']
        instruct = random.choice(INSTRUCTIONS).format(refexp)
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
            bboxes=bboxes)
        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    image = batch[0]['image'].unsqueeze(dim=0)
    bboxes = batch[0]['bboxes']
    return input_ids, image, bboxes


def eval_model(args):
    # Model
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)
    model.config.box_score_thres = args.box_score_thres

    dataset = RefCOCOTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        test_mode=True,
        conv_temp='llava'
    )
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
        sampler=distributed_sampler, collate_fn=custom_collate_fn)

    m_iou = torch.tensor(0.).cuda()
    thres_iou = torch.tensor(0.).cuda()
    invalid = torch.tensor(0.).cuda()

    for input_ids, image, bboxes in dataloader:
        input_ids = input_ids.cuda()
        image = image.cuda()
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=3,
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
        ious = box_iou(
            center_to_corners_format(selected_boxes),
            center_to_corners_format(bboxes)
        )
        ious = torch.max(ious, dim=-1).values

        m_iou += ious[0]
        thres_iou += 1 if ious[0] > args.threshold else 0

    torch.distributed.reduce(thres_iou, dst=0)
    torch.distributed.reduce(m_iou, dst=0)
    torch.distributed.reduce(invalid, dst=0)

    if torch.distributed.get_rank() == 0:
        print(args.ann_file.split('/')[-1])
        print("iou@{} accu: {}".format(args.threshold, thres_iou.item() / len(dataset)))
        print("m_iou: {}".format(m_iou.item() / len(dataset)))
        print("missing percentage: {}".format(invalid.item() / len(dataset)))
        print('=' * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--ann-file", type=str, default="refcoco_val.json")
    parser.add_argument("--img-prefix", type=str, default="coco/train2017")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--box_score_thres", type=float, default=0.15)
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    init_distributed_mode(args)

    eval_model(args)
