import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
from groma.constants import DEFAULT_TOKENS, IGNORE_INDEX
from groma.data.conversation import conv_templates


class LLaVAInstruct(Dataset):
    """Dataset for simple image-text pairs."""

    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default'):
        super(LLaVAInstruct, self).__init__()
        self.meta_data = json.load(open(ann_file, "r"))
        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

    def __len__(self):
        return len(self.meta_data)

    def preprocess(self, conversations):
        new_conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))
        
        assert len(conversations) % 2 == 0
        for i, conversation in enumerate(conversations):
            chat = conversation['value']
            chat = chat.replace('<image>', '')
            chat = chat.replace('\n', ' ')
            if i % 2 == 1:
                chat = DEFAULT_TOKENS['sep'] + chat + DEFAULT_TOKENS['sep']
            new_conversations.append((self.conv_temp.roles[i%2], chat))
        prompt = self.conv_temp.get_prompt(new_conversations)

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
            source='llava'
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_source = self.meta_data[i]
        data_dict = self.preprocess(data_source['conversations'])
        if 'image' in data_source:
            image_file = data_source['image']
            image_folder = self.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB').resize((448, 448))
            image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data_dict['image'] = image
        return data_dict
