# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task4classification.py
@time: 2021/1/25 15:06
"""
import _pickle as pickle
import csv
import os
from argparse import Namespace
from typing import Tuple, Dict, List, NoReturn

import attr
from attr.validators import instance_of
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.utils import to_categorical
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from dependence.bert4keras.snippets import sequence_padding
from dependence.keras_contrib.callbacks import ClsMetrics
from modeling.bert_classification_model import Bert4Classification


@attr.s
class ClsDataset(Dataset):
    root = attr.ib(type=str)
    mode = attr.ib(type=str)
    data = attr.ib(type=list, default=attr.Factory(list), init=False)

    def __attrs_post_init__(self):
        if self.mode == "train":
            with open(os.path.join(self.root, "train.txt"), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    assert len(row) == 3
                    self.data.append(row)
        if self.mode == "test":
            with open(os.path.join(self.root, "test.txt"), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    assert len(row) == 3
                    self.data.append(row)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@attr.s
class Task4Cls:
    args = attr.ib(validator=instance_of(Namespace))
    train_set = attr.ib(type=ClsDataset, init=False)

    def __attrs_post_init__(self):
        vocab = os.path.join(self.args.bert_model_path, "vocab.txt")
        self.tokenizer = BertWordPieceTokenizer(vocab, pad_token="[PAD]")
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
        if not os.path.exists(self.args.output_root):
            os.mkdir(self.args.output_root)
        self.train_set = ClsDataset(root="./resource/")

    def build_label_dict(self, labels) -> Tuple[Dict[str, int], Dict[int, str]]:
        count = 0
        label_set = set(labels)
        label2index = {}
        index2label = {}
        for label_one in label_set:
            label2index[label_one] = count
            index2label[count] = label_one
            count = count + 1
        l2i_path = os.path.join(self.args.output_root, "label2index.pickle")
        with open(l2i_path, "wb") as f:
            pickle.dump(label2index, f)
        i2l_path = os.path.join(self.args.output_root, "index2label.pickle")
        with open(i2l_path, "wb") as g:
            pickle.dump(label2index, g)
        return label2index, index2label

    @staticmethod
    def build_model(args, word_dict, label_dict) -> Model:
        bert = Bert4Classification(args=args, word_dict=word_dict, label_dict=label_dict)
        return bert.build_model()

    def train(self):
        # 处理数据，构建字典
        labels = ["neg", "pos"]
        word_dict = self.tokenizer.get_vocab()
        self.label_dict, self.label_dict_invert = self.build_label_dict(labels)
        # # 构建模型（模型依赖于字典）
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=self.args.batch_size,
                                  collate_fn=self.collate_fn,
                                  shuffle=True,
                                  pin_memory=True)

        test_loader = DataLoader(dataset=self.train_set,
                                 batch_size=self.args.batch_size,
                                 collate_fn=self.collate_fn,
                                 shuffle=True,
                                 pin_memory=True)

        model = self.build_model(self.args, word_dict, self.label_dict)
        #
        # # callback 等回调函数
        call_backs = self.build_callbacks()
        # 训练模型
        call_backs.append(ClsMetrics(validation_data=test_loader))
        # train_generator = self.load_data_set(train_raw, label_dict=self.label_dict)
        model.fit_generator(train_loader.__iter__(),
                            steps_per_epoch=len(train_loader),
                            validation_data=test_loader.__iter__(),
                            validation_steps=len(test_loader),
                            epochs=self.args.epochs,
                            callbacks=call_backs,
                            # use_multiprocessing=True
                            )

    def predict(self, title: str, word: str) -> NoReturn:
        label2index_path = os.path.join(self.args.output_root, "label2index.pickle")
        index2label_path = os.path.join(self.args.output_root, "index2label.pickle")
        with open(label2index_path, 'rb') as f:
            label_dict = pickle.load(f)
        with open(index2label_path, 'rb') as f:
            index_dict = pickle.load(f)
        word_dict = self.tokenizer.get_vocab()
        pre_model = load_model("bert_cls_best.hdf5")
        # 加载自定义词典
        encoder = self.tokenizer.encode(sequence=title, pair=word)
        tokens_ids = encoder.ids
        segment_ids = encoder.type_ids
        inputs = [tokens_ids, segment_ids]
        pre = pre_model.predict(inputs)
        print(pre)

    # def save(self, model, model_name):
    #     output_dir = self.args.output_root
    #     out_prefix = "output_"
    #     out_nodes = []
    #     for i in range(len(model.outputs)):
    #         out_nodes.append(out_prefix + str(i + 1))
    #         tf.identity(model.outputs[i], out_prefix + str(i + 1))
    #     sess = K.get_session()
    #     from tensorflow.python.framework import graph_util, graph_io
    #     init_graph = sess.graph.as_graph_def()
    #     main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    #     graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    @staticmethod
    def convert_idx_to_name(y, id2label, array_indexes):
        y = [[id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    @staticmethod
    def build_callbacks():
        call_backs = []
        tensor_board = TensorBoard(log_dir="./logs", write_graph=False)
        call_backs.append(tensor_board)
        checkpoint = ModelCheckpoint('bert_cls_best.hdf5', monitor='val_acc', verbose=2, save_best_only=True,
                                     mode='max',
                                     save_weights_only=True)
        call_backs.append(checkpoint)
        early_stop = EarlyStopping('val_acc', patience=4, mode='max', verbose=2, restore_best_weights=True)
        call_backs.append(early_stop)

        return call_backs

    @staticmethod
    def load_data(filename: str) -> List:

        D = []
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                assert len(row) == 3
                D.append(row)
        return D

    def collate_fn(self, data):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for di in data:
            doc, word, label = di
            encoder = self.tokenizer.encode(sequence=doc, pair=word)
            tokens_ids = encoder.ids
            segment_ids = encoder.type_ids
            batch_token_ids.append(tokens_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(self.label_dict[label])

        batch_token_ids = sequence_padding(batch_token_ids, value=self.tokenizer.token_to_id('[PAD]'))
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_labels = to_categorical(batch_labels, len(self.label_dict)).astype(int)
        return [batch_token_ids, batch_segment_ids], batch_labels
