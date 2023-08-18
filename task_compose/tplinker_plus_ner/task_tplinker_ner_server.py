
# -----*----coding:utf8-----*----
import argparse
import json
import os
import pickle
import torch
import torch.nn as nn
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import BertTokenizerFast

from datahelper.tplinker_plus.tplinker_plus_dataset import (
    convert_examples_to_features,
    TplinkerPlusNerDataset,
    TplinkerPlusNerInputExample,
    trans_ij2k,
)
from metrics.tplinker_plus_ner.ner_f1 import TplinkerNerF1Metric
from modeling.tplinker_plus.configure_tplinker_plus import TplinkerPlusNerConfig
from modeling.tplinker_plus.modeling_tplinker_plus import TplinkerPlusNer
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI



class TplinkerPlusNerDataModule(pl.LightningDataModule):
    def __init__(
        self, train_data, dev_data, max_length, batch_size, bert_model, num_labels,workers
    ) -> None:
    
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.label_encode = LabelEncoder()
        self.label_encode.fit(self.get_labels())
        assert self.label_encode.classes_.shape[0] == num_labels
        self.mapij2k = {
            (i, j): trans_ij2k(max_length, i, j)
            for i in range(max_length)
            for j in range(max_length)
            if j >= i
        }
        self.mapk2ij = {v: k for k, v in self.mapij2k.items()}
        self.cache_path = os.path.join(os.path.dirname(train_data), "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.train_data = train_data
        self.dev_data = dev_data
        self.max_length = max_length
        self.batch_size = batch_size
        self.workers=workers
        super().__init__()

    def prepare_data(self):
        train_examples = list(self.read_train_data(self.train_data))
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=self.tokenizer,
            label_encode=self.label_encode,
            max_length=self.max_length,
        )
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "wb") as g:
            pickle.dump(train_features, g)

        dev_examples = list(self.read_train_data(self.dev_data))
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=self.tokenizer,
            label_encode=self.label_encode,
            max_length=self.max_length,
        )
        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "wb") as g:
            pickle.dump(dev_features, g)

    @staticmethod
    def read_train_data(path):
        with open(path, "r", encoding="utf-8") as g:
            s = json.loads(g.read())
            data = s["data"]
            name = s["name"]
            example_index = 0
            for index,d in enumerate(data):
                # if index>100:break
                context_text = d["context"]
                labels = []
                for qa in d["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    answers = qa["answers"]
                    if len(answers) > 0:
                        if question == "标题中产品提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "HC", answers[0]["text"]])

                        if question == "标题中品牌提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "HP", answers[0]["text"]])

                        if question == "标题中系列型号提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "XL", answers[0]["text"]])
                
                yield TplinkerPlusNerInputExample(text=context_text, labels=labels)

    def get_labels(self):
        return ["HC", "HP", "XL"]

    def setup(self, stage: str) -> None:
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "rb") as g:
            self.train_features = pickle.load(g)

        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "rb") as g:
            self.dev_features = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(
            dataset=TplinkerPlusNerDataset(self.train_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TplinkerPlusNerDataset(self.dev_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


def main():
    model=TplinkerPlusNer.from_pretrained("saved_model")

    





if __name__ == "__main__":
    main()
