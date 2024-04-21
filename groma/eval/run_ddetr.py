import os
import copy
import torch
import argparse
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from mmcv.ops.nms import nms
from transformers import AutoTokenizer, AutoImageProcessor
from transformers.image_transforms import center_to_corners_format

from groma.model.ddetr import CustomDDETRModel
from groma.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def draw_box(box, image, index, output_dir):
    w, h = image.size
    box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red")
    output_file = os.path.join(output_dir, 'r{}.jpg'.format(index))
    image.save(output_file, "JPEG")
    return


def eval_model(model_name, image_file):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    vis_processor = AutoImageProcessor.from_pretrained(
        'checkpoints/dinov2-large/', do_resize=False, do_center_crop=False)
    model = CustomDDETRModel.from_pretrained(model_name).cuda()

    raw_image = load_image(image_file)
    raw_image = raw_image.resize((448, 448))
    image = vis_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    image = image.unsqueeze(dim=0).to('cuda')

    with torch.inference_mode():
        outputs = model(image)

    pred_boxes = outputs.pred_boxes
    pred_boxes = center_to_corners_format(pred_boxes)
    scores_coco = outputs.logits['coco'].squeeze().sigmoid()
    scores_sa1b = outputs.logits['sa1b'].squeeze().sigmoid()

    # sort_scores_coco = sorted(range(len(scores_coco)), key=lambda k: scores_coco[k])[::-1]
    # sort_scores_sa1b = sorted(range(len(scores_sa1b)), key=lambda k: scores_sa1b[k])[::-1]
    # print(scores_coco[sort_scores_coco[:10]])
    # print(scores_sa1b[sort_scores_sa1b[:10]])
    # print(torch.mean(scores_coco))
    # print(torch.mean(scores_sa1b))
    # print('=============================================')

    nms_inds = nms(pred_boxes[0], scores_coco + scores_sa1b, 0.8)[-1]
    # thres_scores_coco = [i for i in range(len(scores_coco)) if scores_coco[i] >= 0.4 and i in nms_inds]
    # thres_scores_sa1b = [i for i in range(len(scores_sa1b)) if scores_sa1b[i] >= 0.5 and i in nms_inds]
    # thres_scores_comb = list(set(thres_scores_coco + thres_scores_sa1b))
    thres_scores_comb = [i for i in range(len(scores_coco)) if
                         scores_coco[i] ** 0.3 * scores_sa1b[i] ** 0.7 >= 0.4 and i in nms_inds]

    output_dir = 'det_vis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for i, ind in enumerate(sort_score_inds):
    #     print(scores[ind])
    #     img_copy = copy.deepcopy(raw_image)
    #     draw_box(pred_boxes[0, ind, :], img_copy, i, output_dir)

    # img_copy = copy.deepcopy(raw_image)
    # for box in pred_boxes[0, nms_inds]:
    #     w, h = img_copy.size
    #     box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    #     draw = ImageDraw.Draw(img_copy)
    #     draw.rectangle(box, outline="red")
    # img_copy.save('{}/{}_raw.jpg'.format(output_dir, image_file.split('/')[1].split('.')[0]), "JPEG")

    # img_copy = copy.deepcopy(raw_image)
    # for box in pred_boxes[0, thres_scores_coco]:
    #     w, h = img_copy.size
    #     box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    #     draw = ImageDraw.Draw(img_copy)
    #     draw.rectangle(box, outline="red")
    # img_copy.save('{}/filter_coco.jpg'.format(output_dir), "JPEG")

    # img_copy = copy.deepcopy(raw_image)
    # for box in pred_boxes[0, thres_scores_sa1b]:
    #     w, h = img_copy.size
    #     box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
    #     draw = ImageDraw.Draw(img_copy)
    #     draw.rectangle(box, outline="red")
    # img_copy.save('{}/filter_sa1b.jpg'.format(output_dir), "JPEG")

    img_copy = copy.deepcopy(raw_image)
    for box in pred_boxes[0, thres_scores_comb]:
        w, h = img_copy.size
        box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(box, outline="red")
    img_copy.save('{}/{}_filter.jpg'.format(output_dir, image_file.split('/')[1].split('.')[0]), "JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-dir", type=str, required=True)
    args = parser.parse_args()

    model_name = os.path.expanduser(args.model_name)
    image_files = os.listdir(args.image_dir)
    for image_file in image_files:
        image_file = os.path.join(args.image_dir, image_file)
        eval_model(model_name, image_file)
