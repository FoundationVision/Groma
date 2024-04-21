import json
from collections import defaultdict
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('--review-file', default="qa90_review.jsonl")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    review_file = open(args.review_file, 'r')
    scores = defaultdict(list)
    for review_str in review_file:
        review = json.loads(review_str)
        scores[review['category']].append(review['tuple'])
        scores['all'].append(review['tuple'])
    for k, v in scores.items():
        stats = np.asarray(v).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        print(k, stats, round(stats[1]/stats[0]*100, 1))
    print('=================================')
