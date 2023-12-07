from pathlib import Path

import torch
# TODO from google.colab import drive
from torch.utils.data import TensorDataset
from transformers import (
    BertTokenizer,
)

BERT_MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
SEED = 42

WORKDIR = Path(".")
DATADIR = WORKDIR / "data"
MODELDIR = WORKDIR / "model_old"

dataset_path = WORKDIR / 'dataset.csv'
train_dataset_path = DATADIR / 'train_dataset.pt'
test_dataset_path = DATADIR / 'test_dataset.pt'
tokenized_train_dataset_path = DATADIR / 'train_dataset_tokenized.pt'
tokenized_test_dataset_path = DATADIR / 'test_dataset_tokenized.pt'
untransferred_testset_path = DATADIR / 'untransferred_testset.pt'
transferred_testset_path = DATADIR / 'transferred_testset.pt'
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
