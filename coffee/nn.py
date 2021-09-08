from typing import Any, Sequence

import torch.nn as nn
from torch import Tensor
from transformers import AutoModel  # type: ignore

__all__ = ["Model"]


class Model(nn.Module):
    def __init__(self, modelname_or_path: str, config: Any):
        super(Model, self).__init__()
        self.config = config
        self.arch = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None) -> Sequence[Tensor]:

        outputs = self.arch(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
