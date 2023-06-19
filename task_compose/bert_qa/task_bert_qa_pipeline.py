# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer, BertTokenizerFast
from torch import Tensor
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
import logging

log = logging.getLogger()

from transformers import pipeline
"""
针对中文的question answer 进行预测

"""


class BertForQAConfig(BertConfig):

    def __init__(self, **kwargs):
        super(BertForQAConfig, self).__init__(**kwargs)


class BertForQuestionAnswering(BertPreTrainedModel):
    """Finetuning Bert Model for the question answering task."""

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_classifier = nn.Linear(config.hidden_size, 2)
        self.qa_classifier.weight = truncated_normal_(self.qa_classifier.weight, mean=0, std=0.02)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor):
        """
        Args:
            input_ids: Bert input tokens, tensor of shape [batch, seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [batch, seq_len]
            attention_mask: attention mask, tensor of shape [batch, seq_len]
        Returns:
            start_logits: start/non-start logits of shape [batch, seq_len]
            end_logits: end/non-end logits of shape [batch, seq_len]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_heatmap = self.dropout(bert_outputs[0])  # [batch, seq_len, hidden]
        logits = self.qa_classifier(sequence_heatmap)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #         print(start_logits)
        return start_logits, end_logits


if __name__ == "__main__":

    log.info("start:品类识别模型初始化")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bert_config = BertForQAConfig.from_pretrained(
        "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")

    #         self.model.to(self.device)
    #         self.model.eval()

    model = BertForQuestionAnswering(bert_config).from_pretrained("./output", config=bert_config).to(device)
    #         print(self.model.state_dict())
    tokenizer = BertTokenizerFast.from_pretrained(
        "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")
    log.info("end:品类识别模型初始化完成")
    pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    out=pipe(question=["标题中品牌提及有哪些", "标题中产品提及有哪些"],
         context=["理肤泉b5面膜", "海尔电冰箱"],
         handle_impossible_answer=True,
         max_seq_len=128,
         align_to_words=False,
         topk=1)
    print(out)
