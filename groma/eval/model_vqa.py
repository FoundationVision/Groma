import os
import json
import torch
import argparse
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor

from groma.utils import disable_torch_init
from groma.constants import DEFAULT_TOKENS
from groma.model.groma import GromaModel
from groma.data.conversation import conv_templates


class LLaVABench(Dataset):
    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, conv_temp='default'):
        super(LLaVABench, self).__init__()
        self.meta_data = [json.loads(q) for q in open(ann_file, "r")]
        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]

    def __len__(self):
        return len(self.meta_data)

    def preprocess(self, data_source):
        conversations = []
        instruct = "Here is an image with region crops from it. "
        instruct += "Image: {}. ".format(DEFAULT_TOKENS['image'])
        instruct += "Regions: {}.".format(DEFAULT_TOKENS['region'])
        answer = 'Thank you for the image! How can I assist you with it?'
        conversations.append((self.conv_temp.roles[0], instruct))
        conversations.append((self.conv_temp.roles[1], answer))
        conversations.append((self.conv_temp.roles[0], data_source['text']))
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
            question_id=data_source['question_id'],
            category=data_source['category']
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_source = self.meta_data[i]
        data_dict = self.preprocess(data_source)
        image_file = data_source['image']
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB').resize((448, 448))
        image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict['image'] = image
        return data_dict


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    img_processor = AutoImageProcessor.from_pretrained(model_name, do_resize=False, do_center_crop=False)
    model = GromaModel.from_pretrained(model_name).cuda()
    model.init_special_token_id(tokenizer)

    dataset = LLaVABench(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        img_processor=img_processor,
        conv_temp='llava'
    )

    ans_file = open(args.answers_file, "w")
    for sample in dataset:
        input_ids = sample['input_ids'].cuda()
        image = sample['image'].cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image.unsqueeze(0),
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )
        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        result = {
            "question_id": sample["question_id"],
            "text": outputs,
            "category": sample["category"]
        }
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="checkpoints/groma-finetune/")
    parser.add_argument("--ann-file", type=str, default="groma-bench/qa90_questions.jsonl")
    parser.add_argument("--img-prefix", type=str, default="coco/")
    parser.add_argument("--answers-file", type=str, default="qa90_walle_answer.jsonl")
    args = parser.parse_args()

    eval_model(args)

