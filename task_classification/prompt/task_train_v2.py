# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train_v2.py
@time: 2022/5/6 10:31
"""

import argparse
import csv
from abc import ABC
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from pytorch_lightning import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


def read_csv(path, delimiter="\t", quotechar=None):
    with open(path, "r", encoding="utf-8") as g:
        reader = csv.reader(g, delimiter=delimiter, quotechar=quotechar)
        for line in reader:
            yield line


def get_train_examples(path, delimiter="\t", quotechar=None):
    es = []
    t = read_csv(path=path, delimiter=delimiter, quotechar=quotechar)
    guid_num = 0
    for e in t:
        guid = f"train_id_{guid_num}"
        guid_num += 1
        text = f"这是关于{e[0]}的商品,{e[1]}。参数类型为{e[2]},参数值为{e[3]}"
        if e[-1] == "positive":
            label = 1
        else:
            label = 0
        es.append(InputExample(guid=guid, text_a=text, label=label))
    return es


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parser.add_argument("--max_seq_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", choices=["linear", "onecycle", "polydecay"], default="onecycle")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")
    parser.add_argument("--max_epochs", default=1, type=int, help="epochs")
    # argparse解析参数分为模型和数据
    model_parse = parser.add_argument_group(title="model params")
    model_parse.add_argument("--model_name", default="bert", type=str, help="saved model name")
    model_parse.add_argument("--pretrained_model", type=str, help="pretrained model path")
    data_parse = parser.add_argument_group(title="data params")
    data_parse.add_argument("--train_data", type=str, default="", help="train data path")
    data_parse.add_argument("--test_data", type=str, default="", help="test data path")
    data_parse.add_argument("--dev_data", type=str, default="", help="dev data path")

    args = parser.parse_args()
    # 构建模型
    plm, tokenizer, config, wrapper = load_plm(args.model_name, args.pretrained_model)

    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"}。该参数值是否{"mask"}？',
        tokenizer=tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        tokenizer=tokenizer,
        num_classes=2,
        label_words=[["不合理"], ["合理"]],
    )
    print(promptVerbalizer.label_words_ids)

    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    train_dataloader = PromptDataLoader(dataset=get_train_examples(args.train_data),
                                        tokenizer=tokenizer,
                                        tokenizer_wrapper_class=wrapper,
                                        template=promptTemplate,
                                        verbalizer=promptVerbalizer
                                        )
    use_cuda = True

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(args.max_epochs):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
