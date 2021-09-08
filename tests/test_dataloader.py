import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from transformers import AutoTokenizer

from coffee.helpers import make_loader


class Config:
    # tokenizer
    tokenizer_name = "distilbert-base-cased-distilled-squad"
    max_seq_length = 384
    doc_stride = 128

    # dataloader
    train_batch_size = 4
    eval_batch_size = 8


def test_loader():

    args = Config()

    train = pd.read_csv("data/official_data/train.csv")
    external_mlqa = pd.read_csv("data/external_data/mlqa_hindi.csv")
    external_xquad = pd.read_csv("data/external_data/xquad.csv")
    external_train = pd.concat([external_mlqa, external_xquad])

    def create_folds(data, num_splits):
        data["kfold"] = -1
        kf = model_selection.StratifiedKFold(
            n_splits=num_splits, shuffle=True, random_state=2021
        )
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data["language"])):
            data.loc[v_, "kfold"] = f
        return data

    train = create_folds(train, num_splits=5)
    external_train["kfold"] = -1
    external_train["id"] = list(np.arange(1, len(external_train) + 1))
    train = pd.concat([train, external_train]).reset_index(drop=True)

    def convert_answers(row):
        return {"answer_start": [row[0]], "text": [row[1]]}

    train["answers"] = train[["answer_start", "answer_text"]].apply(
        convert_answers, axis=1
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # data loaders
    train_dataloader, valid_dataloader = make_loader(
        args, data=train, tokenizer=tokenizer, fold=1
    )

    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(valid_dataloader, torch.utils.data.DataLoader)
