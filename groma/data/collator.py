import torch
import transformers
from dataclasses import dataclass

from groma.constants import IGNORE_INDEX


@dataclass
class DataCollatorForHybridDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        meta_keys = ('input_ids', 'labels', 'image', 'source')
        input_ids, labels, images, sources = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        refer_boxes = [instance.get('refer_boxes', torch.empty(0, 4)) for instance in instances]
        ground_boxes = [instance.get('ground_boxes', torch.empty(0, 4)) for instance in instances]
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            images=images,
            refer_boxes=refer_boxes,
            ground_boxes=ground_boxes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        return batch


@dataclass
class DataCollatorForDetDataset(object):
    def __call__(self, instances):
        meta_keys = ('image', 'class_labels', 'bboxes', 'source')
        images, class_labels, bboxes, sources = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
        images = torch.stack(images)
        assert len(list(set(sources))) == 1, "data in the same batch should have the same data source."
        labels = [{'class_labels': label, "boxes": bbox, "source": source} for
                  label, bbox, source in zip(class_labels, bboxes, sources)]
        batch = dict(images=images, labels=labels)
        return batch


@dataclass
class DataCollatorForDetEvalDataset(object):
    def __call__(self, instances):
        meta_keys = ('image', 'ori_shape')
        images, ori_shapes = tuple([instance.get(key, None) for instance in instances] for key in meta_keys)
        images = torch.stack(images)
        ori_shapes = torch.stack([torch.tensor(x[:2]) for x in ori_shapes])
        batch = dict(images=images, ori_shapes=ori_shapes)
        return batch
