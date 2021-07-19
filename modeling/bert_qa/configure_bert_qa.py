# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: configure_bert_qa.py
@time: 2021/6/9 20:40
"""

from transformers import BertConfig


class BertForQAConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForQAConfig, self).__init__(**kwargs)
