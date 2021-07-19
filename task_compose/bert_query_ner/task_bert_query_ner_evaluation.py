# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_bert_mrc_evaluation.py
@time: 2021/5/21 11:35
"""

import os
import argparse
from abc import ABCMeta
from typing import Iterator

from pytorch_lightning import Trainer
from task_compose.bert_query_ner.task_bert_query_ner_train import BertLabeling
import json
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader


class InferIterData(IterableDataset, metaclass=ABCMeta):

    def __init__(self, path, tokenizer, max_length):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def patch_worker_data(iterable, number_workers, worker_id, length):
        def get_iter_start_pos(gen):
            start_pos = worker_id * length
            for i in gen:
                if start_pos:
                    start_pos -= 1
                    continue
                yield i

        def filter_elements_per_worker(gen):
            x = length
            y = (number_workers - 1) * length
            for i in gen:
                if x:
                    yield i
                    x -= 1
                else:
                    if y != 1:
                        y -= 1
                        continue
                    else:
                        x = length
                        y = (number_workers - 1) * length

        iterable = iter(iterable)
        iterable = get_iter_start_pos(iterable)
        if number_workers > 1:
            iterable = filter_elements_per_worker(iterable)
        return iterable

    def reader_iter(self):
        with open(self.path, "r", encoding="utf-8") as f:
            d = json.load(f)
            yield from d

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        number_workers = worker_info.num_workers
        worker_id = worker_info.id
        worker_data = self.patch_worker_data(self.reader_iter(), number_workers=number_workers, worker_id=worker_id,
                                             length=10)
        for data in worker_data:
            query = data["query"]
            context = data["context"]
            query_context_tokens = self.tokenizer.encode(query, context, add_special_tokens=True)
            tokens = query_context_tokens.ids
            type_ids = query_context_tokens.type_ids
            offsets = query_context_tokens.offsets
            label_mask = [
                (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
                for token_idx in range(len(tokens))
            ]

            tokens = tokens[: self.max_length]
            type_ids = type_ids[: self.max_length]

            sep_token = self.tokenizer.token_to_id("[SEP]")
            if tokens[-1] != sep_token:
                assert len(tokens) == self.max_length
                tokens = tokens[: -1] + [sep_token]
            return [
                query,
                context,
                torch.LongTensor(tokens),
                torch.LongTensor(type_ids),
                torch.LongTensor(label_mask),
            ]


class InferDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        data = self.all_data[item]
        tokenizer = self.tokenizer
        query = data["query"]
        context = data["context"]
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]

        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]

        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]

        yield [
            query,
            context,
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(label_mask),
        ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

    parser.add_argument("--data_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/gds_associate_project/data_repository/mrc_extract_data/gds_title_theme_keywords",
                        help="data dir")
    parser.add_argument("--bert_config_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
                        help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_epochs", default=1, type=int,
                        help="epochs")

    parser = BertLabeling.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = Trainer(gpus=1)
    model = BertLabeling(args)
    CHECKPOINTS = "./output_dir/epoch=0-step=1249.ckpt"
    hparams_file = "./output_dir/lightning_logs/version_0/hparams.yaml"
    model = model.load_from_checkpoint(
        hparams_file=hparams_file,
        checkpoint_path=CHECKPOINTS)
    # infer_dataset = InferDataset(json_path=os.path.join(args.data_dir, "train.json"),
    #                              tokenizer=model.tokenizer,
    #                              max_length=128
    #                              )
    infer_dataset = InferIterData(path=os.path.join(args.data_dir, "train.json"),
                                  tokenizer=model.tokenizer,
                                  max_length=128)
    dataloasder = DataLoader(infer_dataset)
    data = trainer.predict(model=model, dataloaders=dataloasder)
