

from transformers import BertConfig


class GlobalPointerNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(GlobalPointerNerConfig, self).__init__(**kwargs)
        self.head_size= kwargs.get("head_size", 4)
