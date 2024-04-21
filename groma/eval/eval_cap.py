import os
import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=str, default="refcocog.json")
    parser.add_argument("--result-dir", type=str, default="refcocog_eval_output")
    args = parser.parse_args()

    results = []
    for result_file in os.listdir(args.result_dir):
        if 'all.json' not in result_file:
            results.extend(json.load(open(f"{args.result_dir}/{result_file}", "r")))

    results_map = dict()
    for i, result in enumerate(results):
        key = result['image_id']
        results_map[key] = i
    result_inds = results_map.values()
    results = [results[i] for i in result_inds]

    all_results_file = f"{args.result_dir}/all.json"
    with open(all_results_file, 'w') as f:
        json.dump(results, f)
    
    coco = COCO(args.ann_file)
    coco_result = coco.loadRes(all_results_file)
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    