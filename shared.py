from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from transformers import (
    BertTokenizer,
)
import random
import numpy as np

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BERT_MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
set_seed()

WORKDIR = Path(".")
DATADIR = WORKDIR / "data"
MODELDIR = WORKDIR / "model"

dataset_path = WORKDIR / 'dataset.csv'

train_dataset_path = DATADIR / 'train_dataset.pt'
test_dataset_path = DATADIR / 'test_dataset.pt'

tokenized_train_dataset_path = DATADIR / 'train_dataset_tokenized.pt'
tokenized_test_dataset_path = DATADIR / 'test_dataset_tokenized.pt'

untransferred_fake_testset_path = DATADIR / 'untransferred_fake_testset.pt'
transferred_fake_testset_path = DATADIR / 'transferred_fake_testset.pt'
transferred_fake_csv_path = DATADIR / 'transferred_fake_testset.csv'

untransferred_real_testset_path = DATADIR / 'untransferred_real_testset.pt'
transferred_real_testset_path = DATADIR / 'transferred_real_testset.pt'
transferred_real_csv_path = DATADIR / 'transferred_real_testset.csv'

model_path = MODELDIR / 'BERT_finetuned.ckpt'

def tokenize_texts(df, feature="all_text"):
    input_ids = []
    attention_masks = []

    for text in df[feature]:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids.append(encoded_dict["input_ids"])

        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def create_tensor_dataset(df, feature="all_text"):
    input_ids, attention_masks = tokenize_texts(df, feature=feature)
    labels = torch.tensor(df["label"].values)

    return TensorDataset(input_ids, attention_masks, labels)