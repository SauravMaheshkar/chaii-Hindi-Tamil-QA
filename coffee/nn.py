from typing import Any, Sequence

import torch.nn as nn
from torch import Tensor
from transformers import AutoModel  # type: ignore

__all__ = ["Model"]


class Model(nn.Module):
    def __init__(
        self,
        modelname_or_path: str,
        config: Any,
        output_head_dropout_prob: float = 0.1,
        *args,
        **kwargs
    ):
        super(Model, self).__init__(*args, **kwargs)
        self.config = config

        self.arch = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.dropout = nn.Dropout(output_head_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None) -> Sequence[Tensor]:

        # Forward Pass through the Model Architecture
        outputs = self.arch(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Apply Dropout
        vector = self.dropout(sequence_output)

        # Forward Pass through Output Linear Layer
        qa_logits = self.qa_outputs(vector)

        # Output Processing
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
