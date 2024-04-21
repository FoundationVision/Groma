import torch
from torch.utils.data import Sampler
from transformers import Trainer
from typing import List, Iterator, Optional, Sized
from transformers.trainer import (
    has_length,
    get_parameter_names,
    is_sagemaker_mp_enabled,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
)


class RandomBatchSampler(Sampler):
    data_source: Sized
    replacement: bool

    def __init__(
            self,
            data_source: Sized,
            batch_size: int,
            dataset_sizes: List[int],
            generator=None
    ) -> None:

        self.data_source = data_source
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.generator = generator

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        indices = torch.arange(n)
        chunked_indices = torch.split(indices, self.dataset_sizes)
        # sample-level permutation in each dataset
        inner_perm_indices = [x[torch.randperm(len(x), generator=generator)] for x in chunked_indices]
        inner_perm_indices = [x[:len(x) - len(x) % self.batch_size] for x in inner_perm_indices]
        # split into batches
        outer_perm_indices = [torch.split(x, self.batch_size) for x in inner_perm_indices]
        outer_perm_indices = [y for x in outer_perm_indices for y in x]
        # batch-level permutation
        outer_perm_indices = [outer_perm_indices[i] for i in
                              torch.randperm(len(outer_perm_indices), generator=generator)]
        outer_perm_indices = torch.cat(outer_perm_indices, dim=0)
        yield from outer_perm_indices.tolist()

    def __len__(self) -> int:
        return self.num_samples


class GromaTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_data_source:
            cumu_sizes = self.train_dataset.cumulative_sizes
            dataset_sizes = [cumu_sizes[0]] + [cumu_sizes[i] - cumu_sizes[i - 1] for i in range(1, len(cumu_sizes))]
            return RandomBatchSampler(
                self.train_dataset,
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset_sizes
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.use_custom_lr:
                custom_parameters = [name for name, _ in opt_model.named_parameters() if
                                     any(param in name for param in self.args.custom_lr_params)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and n not in custom_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and n not in custom_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and n in custom_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.custom_lr,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and n in custom_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                        "lr": self.args.custom_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if
                                   (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer