import multiprocessing
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = ["loss_fn", "set_seed", "AverageMeter", "optimal_num_of_loader_workers"]


def loss_fn(preds: Any, labels: Any) -> Any:
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def optimal_num_of_loader_workers() -> int:
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus * 4) if num_gpus else num_cpus - 1
    return optimal_value


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val: Any, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # type: ignore
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val
