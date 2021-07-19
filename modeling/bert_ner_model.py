#!/usr/bin/env python
# -*- encoding=utf-8 -*-
"""
@author: 18073701
@email:  18073701@suning.com
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@software: pycharm
@file: bert4ner_model.py
@time: 2020/3/7 12:35 下午
"""
from dependence.bert4keras.layers import *
from dependence.bert4keras.models import build_transformer_model
from dependence.bert4keras.optimizers import Adam
from keras.models import Model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy


class Bert4NerBaseModel(object):
    def __init__(self, args, word_dict, label_dict):
        self.args = args
        # 模型结构参数
        self.word_dict = word_dict
        self.label_dict = label_dict
        # 字典

    def build_model(self):
        model = build_transformer_model(
            self.args.bert_config_path,
            self.args.bert_checkpoint_path,
        )
        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.args.bert_layers - 1)
        output = model.get_layer(output_layer).output
        output = Dense(len(self.label_dict))(output)
        # 构建模型时候[CLS]向量
        CRFlayer = CRF(len(self.label_dict), sparse_target=False, name='pred_ids')
        output = CRFlayer(output)
        # todo 添加训练的callback

        model = Model(model.input, output)
        model.compile(loss=crf_loss,
                      optimizer=Adam(self.args.lr),
                      metrics=[crf_accuracy])
        return model


class Bert4NerAppendCategoryModel:
    def __init__(self, args, word_dict, label_dict):
        self.args = args
        # 模型结构参数
        self.word_dict = word_dict
        self.label_dict = label_dict
        # 字典

    def build_model(self):
        # 构建预训练的bert模型，模型输入保持不变
        model = build_transformer_model(
            self.args.bert_config_path,
            self.args.bert_checkpoint_path,
        )

        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.args.bert_layers - 1)

        output = model.get_layer(output_layer).output
        # TODO 需要对模型的输出进行一个截断，序列标注的时候移除品类词
        output = Dense(len(self.label_dict))(output)
        # 构建模型时候[CLS]向量
        CRFlayer = CRF(len(self.label_dict), sparse_target=False, name='pred_ids')
        output = CRFlayer(output)
        # todo 添加训练的callback

        model = Model(model.input, output)
        model.compile(loss=crf_loss,
                      optimizer=Adam(self.args.lr),
                      metrics=[crf_accuracy])
        return model


class BilstmWithAttention:
    def __init__(self, args, word_dict, label_dict):
        self.args = args
        # 模型结构参数
        self.word_dict = word_dict
        self.label_dict = label_dict
        # 字典

    def build_model(self):
        # 构建预训练的bert模型，模型输入保持不变
        model = build_transformer_model(
            self.args.bert_config_path,
            self.args.bert_checkpoint_path,
        )
        for layer in model.layers[:]:
            layer.trainable = False

        seg_inputs = Input(batch_shape=(None, None), dtype='int32', name='segment_ids')
        # 设置分词特征
        seg_embeddings = Embedding(input_dim=4,
                                   output_dim=self.args.seg_embedding_dim,
                                   mask_zero=True,
                                   embeddings_regularizer='l2',
                                   name='seg_embedding')(seg_inputs)

        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.args.bert_layers - 1)
        output = model.get_layer(output_layer).output
        hidden_dim = output.shape[-1].value
        [_, segment] = model.inputs
        first_seq = Lambda(lambda x: K.cast(K.equal(x, 0), 'int32'))(segment)
        first_seq_len = Lambda(lambda x: K.sum(x[0, :]))(first_seq)
        second_seq = Lambda(lambda x: K.cast(K.equal(x, 1), "int32"))(segment)
        second_seq_len = Lambda(lambda x: K.sum(x[0, :]))(second_seq)
        output = Lambda(lambda x: x[0][:, :K.cast(x[1], "int32"), :])([output, first_seq_len])
        context = Lambda(lambda x: x[0][:, -K.cast(x[1], "int32"):, :])([output, second_seq_len])
        context = Conv1D(filters=100, kernel_size=2, padding='valid')(context)
        context = GlobalMaxPooling1D()(context)
        output = Dropout(rate=self.args.dropout_rate)(output)
        # attention 层进一步构建两者之间的依赖关系
        output = AttentionLayer(return_attention=False)([output, context])
        output = Concatenate()([output, seg_embeddings])
        # 构建模型时候[CLS]向量
        output = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=self.args.dropout_rate))(
            output)

        output = TimeDistributed(Dense(len(self.label_dict)))(output)
        CRFlayer = CRF(len(self.label_dict), sparse_target=False, name='pred_ids')
        output = CRFlayer(output)
        # todo 添加训练的callback
        inputs = model.input
        inputs.append(seg_inputs)
        model = Model(inputs, output)
        model.compile(loss=crf_loss,
                      optimizer=Adam(self.args.lr),
                      metrics=[crf_accuracy])
        return model
