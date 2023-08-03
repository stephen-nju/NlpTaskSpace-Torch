# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
"""

from transformers import BertConfig
class TplinkerPlusNerConfig(BertConfig):
    """
    扩充bert config的配置文件
    """
    def __init__(self, **kwargs):
        super(TplinkerPlusNerConfig, self).__init__(**kwargs)
        self.tok_pair_sample_rate= kwargs.get("tok_pair_sample_rate", 1)
