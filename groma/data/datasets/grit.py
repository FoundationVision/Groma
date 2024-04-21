from multiprocessing.connection import answer_challenge
import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh

from groma.constants import DEFAULT_TOKENS, IGNORE_INDEX
from groma.data.conversation import conv_templates


INSTRUCTIONS = [
    "Locate {} in the image.",
    "Can you spot {} in the photograph?",
    "Identify where {} is located in the picture.",
    "Please detect {} in the picture.",
    "Which region matches the description {}?",
    "Please identify the object that corresponds to {}."
]


class Grit(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default'):
        super(Grit, self).__init__()
        self.meta_data = json.load(open(ann_file, "r"))
        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

    def __len__(self):
        return len(self.meta_data)

    def preprocess(self, data_item):
        template = random.choice(data_item['ref_exps'])
        caption = data_item['caption']
        label = caption[template[0]: template[1]]
        bboxes = [template[2:6]]
        for ref_exp in data_item['ref_exps']:
            if ref_exp[:2] == template[:2] and ref_exp != template:
                bboxes.append(ref_exp[2:6])
        bboxes = torch.tensor(bboxes)
        bboxes = bbox_xyxy_to_cxcywh(bboxes)

        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))

        refexp = DEFAULT_TOKENS['boe'] + label.strip() + DEFAULT_TOKENS['eoe']
        instruct = random.choice(INSTRUCTIONS).format(refexp)
        answer = DEFAULT_TOKENS['sep']
        answer += DEFAULT_TOKENS['bor'] + DEFAULT_TOKENS['gbox'] * len(bboxes) + DEFAULT_TOKENS['eor']
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
            source='grit',
            ground_boxes=bboxes
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_item = self.meta_data[i]
        image_file = data_item['filename']
        image_folder = self.image_folder
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB').resize((448, 448))
        except:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)
        image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict = self.preprocess(data_item)
        data_dict['image'] = image
        return data_dict
