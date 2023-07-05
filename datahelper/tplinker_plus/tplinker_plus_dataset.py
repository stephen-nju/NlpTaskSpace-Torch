# ---# -----*----coding:utf8-----*----

import logging
from functools import partial
from multiprocessing import cpu_count, Pool
from multiprocessing.dummy import Pool
from os import cpu_count

import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


def trans_ij2k(seq_len, i, j):
    '''把第i行，第j列转化成上三角flat后的序号
    '''
    if (i > seq_len - 1) or (j > seq_len - 1) or (i > j):
        return 0
    return int(0.5 * (2 * seq_len - i + 1) * i + (j - i))


def convert_single_example(example, tokenizer, label_encode, max_length):
    encode_inputs = tokenizer(example.text,
                              max_length=max_length,
                              return_offsets_mapping=True,  # 指定改参数，返回切分后的token在文章中的位置
                              return_token_type_ids=True,
                              return_attention_mask=True,
                              truncation=True,
                              padding="max_length"
                              )
    labels = example.labels
    input_ids = encode_inputs.input_ids
    type_ids = encode_inputs.token_type_ids
    offsets = encode_inputs.offset_mapping
    attention_mask = encode_inputs.attention_mask
    # 重新计算每个entiy的start，end位置

    pair_length = max_length * (max_length + 1) // 2
    label_matrix = np.zeros(shape=(pair_length, len(label_encode.classes_)), dtype=int)
    for (start, end, label, entity) in labels:
        # 这里由于是单条数据，利用offsets_mapping 需要跳过特殊字符
        # 还需要注意子词的切分offsets的含义表示该token在原始数据中（start,end）
        token_start_index = 0
        token_end_index = len(input_ids) - 1

        # 从后去除padding和特殊字符的位置
        while offsets[token_end_index][0] == offsets[token_end_index][1] == 0:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start) and (offsets[token_end_index][1] >= end):
            # 无法定位实体的位置
            logging.warning("无法定位实体位置,直接跳过")
            continue
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start:
            token_start_index += 1
        start_position = token_start_index - 1
        while token_end_index > 0 and offsets[token_end_index][1] >= end:
            token_end_index -= 1
        end_position = token_end_index + 1
        label_id = label_encode.transform([label])[0]
        index = trans_ij2k(max_length, start_position, end_position)
        label_matrix[index, label_id] = 1

    return TplinkerPlusNerInputFeature(
            input_ids=input_ids,
            token_type_ids=type_ids,
            attention_mask=attention_mask,
            labels=label_matrix
    )


def convert_examples_to_features(examples, tokenizer, label_encode, max_length, threads=4):
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
                convert_single_example,
                tokenizer=tokenizer,
                label_encode=label_encode,
                max_length=max_length
        )
        features = list(
                tqdm(
                        p.imap(annotate_, examples, chunksize=32),
                        total=len(examples),
                        desc="convert examples to features",
                )
        )
    return features


class TplinkerPlusNerInputExample():
    def __init__(self, text, labels) -> None:
        self.text = text
        self.labels = labels


class TplinkerPlusNerInputFeature():

    def __init__(self, input_ids, token_type_ids, attention_mask, labels) -> None:
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels


class TplinkerPlusNerDataset(Dataset):

    def __init__(self, features) -> None:
        self.features = features
        super().__init__()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]

        return {"input_ids"     : feature.input_ids,
                "token_type_ids": feature.token_type_ids,
                "attention_mask": feature.attention_mask,
                "labels"        : feature.labels

                }
