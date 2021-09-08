from typing import Sequence

from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (  # type: ignore
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from .data_utils import prepare_train_features
from .dataloader import Dataset
from .nn import Model
from .utils import optimal_num_of_loader_workers

__all__ = ["make_model", "make_optimizer", "make_scheduler", "make_loader"]


def make_model(args) -> Sequence:
    config = AutoConfig.from_pretrained(args.config_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = Model(args.model_name_or_path, config=config)
    return config, tokenizer, model


def make_optimizer(args, model) -> Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.epsilon,
        correct_bias=True,
    )

    return optimizer


def make_scheduler(args, optimizer, num_warmup_steps, num_training_steps):
    if args.decay_name == "cosine-warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    return scheduler


def make_loader(args, data, tokenizer, fold) -> Sequence[DataLoader]:
    train_set, valid_set = data[data["kfold"] != fold], data[data["kfold"] == fold]

    train_features, valid_features = [[] for _ in range(2)]
    for i, row in train_set.iterrows():
        train_features += prepare_train_features(args, row, tokenizer)
    for i, row in valid_set.iterrows():
        valid_features += prepare_train_features(args, row, tokenizer)

    train_dataset = Dataset(train_features)
    valid_dataset = Dataset(valid_features)
    print(
        f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}"  # noqa: E501
    )

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        num_workers=optimal_num_of_loader_workers(),
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, valid_dataloader
