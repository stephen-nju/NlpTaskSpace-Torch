# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: mrc_dataset.py
@time: 2021/5/18 14:47
"""
# encoding: utf-8


import json
from abc import ABC, ABCMeta
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset


def collate_to_max_length(batch):
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    return output


class MRCNERDataset(Dataset):

    def __init__(self, json_path, tokenizer, max_length: int = 128, possible_only=False,
                 pad_to_maxlen=False):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        data = self.all_data[item]
        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]
        # end_positions = [x + 1 for x in end_positions]

        query_context_tokens = self.tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # query 先跳过
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # [CLS]  [SEP] 先跳过
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        # 筛选不合适的数据
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # 最大长度截断
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # 句尾添加 [SEP]
        sep_token = self.tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(start_labels, dtype=torch.long),
            torch.tensor(end_labels, dtype=torch.long),
            torch.tensor(start_label_mask, dtype=torch.long),
            torch.tensor(end_label_mask, dtype=torch.long),
            match_labels,
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""

    def __init__(self, dataset: Dataset, max_num: int = 100):
        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    def __getattr__(self, item):
        """other dataset func"""
        return getattr(self.dataset, item)


