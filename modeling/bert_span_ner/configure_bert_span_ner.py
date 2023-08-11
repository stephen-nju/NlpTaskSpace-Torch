
# -*- coding: utf-8 -*-


from transformers import BertConfig


class BertSpanNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForQAConfig, self).__init__(**kwargs)
