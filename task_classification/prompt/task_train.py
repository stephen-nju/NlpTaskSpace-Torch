# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:18073701@cnsuning.com
@license: Apache Licence
@software:PyCharm
@file: task_train.py
@time: 2022/3/30 17:41

"""
import argparse
import csv
from abc import ABC
from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from pytorch_lightning import Trainer
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


class PromptDataModule(pl.LightningDataModule):

    def __init__(self, args: argparse.Namespace, tokenizer, template, verbalizer, tokenizer_wrapper):
        super().__init__()
        self.tokenizer = tokenizer
        self.template = template
        self.verbalizer = verbalizer
        self.tokenizer_wrapper = tokenizer_wrapper
        self.args = args

    @staticmethod
    def read_csv(path, delimiter="\t", quotechar=None):
        with open(path, "r", encoding="utf-8") as g:
            reader = csv.reader(g, delimiter=delimiter, quotechar=quotechar)
            for line in reader:
                yield line

    def get_train_examples(self, path, delimiter="\t", quotechar=None):
        es = []
        t = self.read_csv(path=path, delimiter=delimiter, quotechar=quotechar)
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

    def train_dataloader(self):
        loader = PromptDataLoader(dataset=self.get_train_examples(self.args.train_data),
                                  tokenizer=self.tokenizer,
                                  tokenizer_wrapper_class=self.tokenizer_wrapper,
                                  template=self.template,
                                  verbalizer=self.verbalizer,
                                  batch_size=args.batch_size
                                  )
        return loader



class PromptClassificationModule(pl.LightningModule, ABC):

    def __init__(self, args: argparse.Namespace, model):
        super(PromptClassificationModule, self).__init__()
        self.args = args
        self.model = model
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.args.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches
        self.t_total = (len(train_loader) // tb_size) // ab_size * int(self.trainer.max_epochs)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        warmup_steps = int(self.args.warmup_proportion * self.t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps / self.t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=self.t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.t_total
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, self.t_total,
                                                                  lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        labels = batch['label']
        loss = F.cross_entropy(logits, labels)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parser.add_argument("--max_seq_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", choices=["linear", "onecycle", "polydecay"], default="linear")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")
    # argparse解析参数分为模型和数据
    model_parse = parser.add_argument_group(title="model params")
    model_parse.add_argument("--model_name", default="bert", type=str, help="saved model name")
    model_parse.add_argument("--pretrained_model", type=str, help="pretrained model path")
    data_parse = parser.add_argument_group(title="data params")
    data_parse.add_argument("--train_data", type=str, default="", help="train data path")
    data_parse.add_argument("--test_data", type=str, default="", help="test data path")
    data_parse.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser = Trainer.add_argparse_args(parser)
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

    trainer = Trainer.from_argparse_args(args, logger=False)
    dataModule = PromptDataModule(args=args,
                                  tokenizer=tokenizer,
                                  tokenizer_wrapper=wrapper,
                                  template=promptTemplate,
                                  verbalizer=promptVerbalizer,
                                  )

    model = PromptClassificationModule(args=args,
                                       model=promptModel
                                       )

    trainer.fit(model, dataModule)
