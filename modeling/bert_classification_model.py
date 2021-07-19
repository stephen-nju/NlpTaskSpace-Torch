# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: bert4cls_model.py
@time: 2021/1/26 10:34
"""
import os
from bert4keras.layers import *
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.models import Model
from keras_contrib.optimizers import AdamWarmup
from keras.metrics import Recall,Precision,Accuracy


class Bert4Classification(object):
    def __init__(self, args, word_dict, label_dict):
        self.args = args
        # 模型结构参数
        self.word_dict = word_dict
        self.label_dict = label_dict
        # 字典

    def build_model(self) -> Model:
        if self.args.bert_model_path:
            config_path = os.path.join(self.args.bert_model_path, "bert_config.json")
            checkpoint_path = os.path.join(self.args.bert_model_path, "bert_model.ckpt")
            model = build_transformer_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
            output_layer = 'Transformer-%s-FeedForward-Norm' % (self.args.bert_layers - 1)
            output = model.get_layer(output_layer).output
            output = Lambda(lambda x: x[:, 0:1, :])(output)  # 获取CLS
            output = Lambda(lambda x: K.squeeze(x, axis=1))(output)
            output = Dense(len(self.label_dict), activation="softmax")(output)

            new_model = Model(model.input, output)
            new_model.compile(loss='categorical_crossentropy',
                              optimizer=Adam(self.args.lr),
                              metrics=['accuracy', keras.metrics.Precision()]
                              )
            return new_model
