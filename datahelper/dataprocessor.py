#!/usr/bin/env python
# -*- encoding=utf-8 -*-
"""
@author: 18073701
@email:  18073701@suning.com
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@software: pycharm
@file: dataprocessor.py
@time: 2020/3/7 3:46 下午
"""
import attr
import numpy as np
from attr.validators import instance_of
from tensorflow.keras.utils import to_categorical
from tokenizers.implementations import BaseTokenizer, BertWordPieceTokenizer

from dependence.bert4keras.snippets import sequence_padding


@attr.s
class DataGenerator(object):
    """数据生成器模版
    """
    data = attr.ib()
    batch_size = attr.ib(type=int)

    def __attrs_post_init__(self):
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


@attr.s
class ClsDataGenerator(DataGenerator):
    data = attr.ib()
    label_dict = attr.ib(type=dict)
    tokenizer = attr.ib(type=BaseTokenizer, validator=instance_of(BertWordPieceTokenizer))
    batch_size = attr.ib(type=int, default=32)
    maxlen = attr.ib(type=int, default=50)

    def __iter__(self, random=True):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            doc, word, label = self.data[i]
            encoder = self.tokenizer.encode(sequence=doc, pair=word)
            tokens_ids = encoder.ids
            segment_ids = encoder.type_ids
            batch_token_ids.append(tokens_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(self.label_dict[label])

            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids, padding=self.tokenizer.token_to_id('[PAD]'))
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NerDataGenerator(DataGenerator):
    def __init__(self, data, batch_size, tokenizer, maxlen, label_dict, label_pad_token):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.label_pad_token = label_pad_token
        # 不同的task_name 使用不同的数据处理模式
        super(NerDataGenerator, self).__init__(data, batch_size)

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            token_ids, labels = [self.tokenizer._token_start_id], [self.label_dict.get("O")]
            # CLS 默认标记为O
            for w, l in self.data[i]:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [self.label_dict.get("O")] * len(w_token_ids)
                    else:
                        B = self.label_dict.get("B-" + l)
                        I = self.label_dict.get("I-" + l)
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [self.tokenizer._token_sep_id]
            labels += [self.label_dict.get("O")]
            # sep 默认标记为O
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids, padding=self.tokenizer._token_pad_id)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, padding=self.label_dict.get(self.label_pad_token))
                batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NerDataGeneratorAppendCategory(DataGenerator):
    def __init__(self, data, batch_size, tokenizer, maxlen, label_dict, label_pad_token):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.label_pad_token = label_pad_token
        # 不同的task_name 使用不同的数据处理模式
        super(NerDataGeneratorAppendCategory, self).__init__(data, batch_size)

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            token_ids, labels = [self.tokenizer._token_start_id], [
                self.label_dict.get("O")]
            # CLS 默认标记为O
            for w, l in self.data[i]:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    if l == 'O':
                        token_ids += w_token_ids
                        labels += [self.label_dict.get("O")] * len(w_token_ids)
                    elif l == "category":
                        token_ids += [self.tokenizer._token_end_id]
                        token_ids += w_token_ids
                        labels += [self.label_dict.get("O") * (len(w_token_ids) + 1)]
                    else:
                        token_ids += w_token_ids
                        B = self.label_dict.get("B-" + l)
                        I = self.label_dict.get("I-" + l)
                        if B is None or I is None:
                            raise KeyError("keyError in label dict")
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break

            token_ids += [self.tokenizer._token_end_id]
            labels += [self.label_dict.get("O")]
            # sep 默认标记为O
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids, padding=self.tokenizer._token_pad_id)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, padding=self.label_dict.get(self.label_pad_token))
                batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NerDataGenerator4Attention(DataGenerator):
    def __init__(self, data, batch_size, tokenizer, maxlen, label_dict, label_pad_token):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.label_pad_token = label_pad_token
        # 不同的task_name 使用不同的数据处理模式
        super(NerDataGenerator4Attention, self).__init__(data, batch_size)

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels, batch_category_ids, batch_segment_next, batch_boundary_ids = [], [], [], [], [], []
        for i in idxs:
            token_ids, category_ids, labels, boundary_ids = [self.tokenizer._token_start_id], [], [
                self.label_dict.get("O")], [0]
            # CLS 默认标记为O
            for w, s, l in self.data[i]:
                w_token_ids = self.tokenizer.encode(list(w))[0]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    if l == 'O':
                        token_ids += w_token_ids
                        labels += [self.label_dict.get("O")] * len(w_token_ids)
                        boundary_ids += list(s)
                        assert len(labels) == len(boundary_ids)
                    elif l == "category":
                        category_ids += w_token_ids
                    else:
                        token_ids += w_token_ids
                        boundary_ids += list(s)
                        B = self.label_dict.get("B-" + l)
                        I = self.label_dict.get("I-" + l)
                        if B is None or I is None:
                            raise KeyError("keyError in label dict")
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break

            token_ids += [self.tokenizer._token_end_id]
            boundary_ids += [0]
            labels += [self.label_dict.get("O")]
            # sep 默认标记为O

            segment_ids = [0] * len(token_ids)
            if len(category_ids) == 0:
                raise Exception("sentence missing category")
            segment_ids_next = [1] * len(category_ids)

            batch_boundary_ids.append(boundary_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            batch_category_ids.append(category_ids)
            batch_segment_next.append(segment_ids_next)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids, padding=self.tokenizer._token_pad_id)
                batch_boundary_ids = sequence_padding(batch_boundary_ids, padding=0)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, padding=self.label_dict.get(self.label_pad_token))
                batch_category_ids = sequence_padding(batch_category_ids, padding=self.tokenizer._token_pad_id)
                batch_segment_next = sequence_padding(batch_segment_next, padding=1)
                batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)

                yield [np.concatenate([batch_token_ids, batch_category_ids], axis=-1),
                       np.concatenate([batch_segment_ids, batch_segment_next], axis=-1),
                       batch_boundary_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels, batch_category_ids, batch_segment_next, batch_boundary_ids = [], [], [], [], [], []


class NerDataGeneratorWithPos(DataGenerator):
    def __init__(self, data, batch_size, tokenizer, maxlen, label_dict, label_pad_token, pos_dict, pos_pad_token):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.label_pad_token = label_pad_token
        self.pos_dict = pos_dict
        self.pos_pad_token = pos_pad_token
        super(NerDataGeneratorWithPos, self).__init__(data, batch_size)

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels, batch_category_ids, batch_segment_next, batch_pos_ids, \
        batch_mask_id = [], [], [], [], [], [], []
        for i in idxs:
            # get的时候出现None
            token_ids, labels, pos_ids, mask_ids = [self.tokenizer._token_start_id], [
                self.label_dict.get("O")], [self.pos_dict.get(self.pos_pad_token)], [1]
            # CLS 默认标记为O
            for (w, l, p, m) in zip(self.data[i][0], self.data[i][1], self.data[i][2], self.data[i][3]):
                w_token_ids = self.tokenizer.encode(w)[0]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    L = self.label_dict.get(l)
                    P = self.pos_dict.get(p)
                    if L is None:
                        raise KeyError("key error in label dict")
                    if P is None:
                        raise KeyError("key error in pos dict")
                    labels += [L]
                    pos_ids += [P]
                    mask_ids += [m]

            token_ids += [self.tokenizer._token_end_id]
            labels += [self.label_dict.get("O")]
            pos_ids += [self.pos_dict.get(self.pos_pad_token)]
            mask_ids += [1]
            segment_ids = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            batch_pos_ids.append(pos_ids)
            batch_mask_id.append(mask_ids)
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids, padding=self.tokenizer._token_pad_id)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, padding=self.label_dict.get(self.label_pad_token))
                batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)
                batch_pos_ids = sequence_padding(batch_pos_ids, padding=self.pos_dict.get(self.pos_pad_token))
                batch_mask_id = sequence_padding(batch_mask_id, padding=1)
                yield [batch_token_ids, batch_segment_ids, batch_pos_ids, batch_mask_id], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels, batch_pos_ids, batch_mask_id = [], [], [], [], []
