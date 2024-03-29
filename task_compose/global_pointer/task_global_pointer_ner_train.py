# -----coding=utf-8----

import argparse
import os
import pickle

import json

import torch
import torch.nn as nn
from metrics.global_pointer.f1 import GlobalPointerF1Metric
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import BertTokenizerFast
from typing import Optional
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from datahelper.global_pointer.global_pointer_dataset import (
    GlobalPointerNerDataset,
    GlobalPointerNerInputExample,
    convert_examples_to_features,
)
from modeling.global_pointer.configure_global_pointer import GlobalPointerNerConfig
from modeling.global_pointer.modeling_global_pointer import GlobalPointerNer


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """y_true ([Tensor]): [..., num_classes]
        y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat(
            [y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1
        )
        y_pred_neg = torch.cat(
            [y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1
        )

        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        # return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        return (pos_loss + neg_loss).mean()


class GlobalPointerNerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_path,
        train_data,
        dev_data,
        batch_size,
        max_length,
        num_labels,
        workers,
    ) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.label_encode = LabelEncoder()
        self.label_encode.fit(self.get_labels())
        assert self.label_encode.classes_.shape[0] == num_labels
        self.cache_path = os.path.join(os.path.dirname(train_data), "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        super().__init__()

    def prepare_data(self):
        train_examples = list(self.read_train_data(self.train_data))
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=self.tokenizer,
            label_encode=self.label_encode,
            max_length=self.max_length,
        )
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "wb") as g:
            pickle.dump(train_features, g)

        dev_examples = list(self.read_train_data(self.dev_data))
        dev_features = convert_examples_to_features(
            examples=dev_examples,
            tokenizer=self.tokenizer,
            label_encode=self.label_encode,
            max_length=self.max_length,
        )
        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "wb") as g:
            pickle.dump(dev_features, g)

    @staticmethod
    def read_train_data(path):
        with open(path, "r", encoding="utf-8") as g:
            s = json.loads(g.read())
            data = s["data"]
            name = s["name"]
            example_index = 0
            for index, d in enumerate(data):
                # if index > 100:
                #     break
                context_text = d["context"]
                labels = []
                for qa in d["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    answers = qa["answers"]
                    if len(answers) > 0:
                        if question == "标题中产品提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "HC", answers[0]["text"]])

                        if question == "标题中品牌提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "HP", answers[0]["text"]])

                        if question == "标题中系列型号提及有哪些":
                            start = answers[0]["answer_start"]
                            end = start + len(answers[0]["text"])
                            labels.append([start, end, "XL", answers[0]["text"]])

                yield GlobalPointerNerInputExample(text=context_text, labels=labels)

    def get_labels(self):
        return ["HC", "HP", "XL"]

    def setup(self, stage: str) -> None:
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "rb") as g:
            self.train_features = pickle.load(g)

        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "rb") as g:
            self.dev_features = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(
            dataset=GlobalPointerNerDataset(self.train_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=GlobalPointerNerDataset(self.train_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class GlobalPointerNerModule(pl.LightningModule):
    def __init__(
        self,
        model_path,
        max_length,
        num_labels,
        # opitmizer and lr_scheduler
        optimizer,
        lr,
        lr_scheduler,
        loss_type,
        weight_decay,
        adam_epsilon,
        warmup_proportion,
        final_div_factor,
        rewarm_epoch_num,
    ):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.final_div_factor = final_div_factor
        self.rewarm_epoch_num = rewarm_epoch_num
        self.adam_epsilon = adam_epsilon

        config = GlobalPointerNerConfig.from_pretrained(model_path)
        config.num_labels = num_labels
        self.model = GlobalPointerNer.from_pretrained(model_path, config=config)
        self.optimizer = optimizer
        self.loss = MultilabelCategoricalCrossentropy()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        label_encode = self.trainer.datamodule.label_encode
        self.metric = GlobalPointerF1Metric(label_encode=label_encode)

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.lr,
            )
        elif self.optimizer == "adamw":
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(0.9, 0.999),
                lr=self.lr,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError("optimizer not support")

        num_gpus = self.trainer.num_devices
        # num_gpus = len([x for x in str(self.gpus).split(",") if x.strip()])
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        t_total = (
            len(self.trainer.datamodule.train_dataloader())
            // (self.trainer.accumulate_grad_batches * num_gpus)
            + 1
        ) * self.trainer.max_epochs
        warmup_steps = int(self.warmup_proportion * t_total)
        if self.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                pct_start=float(warmup_steps / t_total),
                final_div_factor=self.final_div_factor,
                total_steps=t_total,
                anneal_strategy="linear",
            )
        elif self.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, warmup_steps, t_total, lr_end=self.lr / 4.0
            )
        elif self.lr_scheduler == "cawr":
            step = (
                len(self.trainer.datamodule.train_dataloader())
                // (self.trainer.accumulate_grad_batches * num_gpus)
                + 1
            ) * self.rewarm_epoch_num
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, step, 1
            )
        else:
            raise ValueError("lr_scheduler does not exist.")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

    def compute_loss(self, logits, targets):
        # print(f"target_shape={targets.shape}")
        # print(f"logits_shape={logits.shape}")
        # matrix length
        targets = torch.reshape(targets, (targets.shape[0]*targets.shape[1],-1))
        logits = torch.reshape(logits, (logits.shape[0]*logits.shape[1],-1))

        return self.loss(logits, targets)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        label = batch["labels"]
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        total_loss = self.compute_loss(logits, label)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        total_loss = self.compute_loss(logits, labels)
        self.log("valid_loss", total_loss, prog_bar=True)

        self.metric.update(logits, labels)

    def on_validation_epoch_end(self):
        p, r, f1 = self.metric.compute(self.global_rank)
        self.metric.reset()


class GlobalPointerNerLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.model_path", "data.model_path")
        parser.link_arguments("data.num_labels", "model.num_labels")
        parser.link_arguments("data.max_length", "model.max_length")


def main():
    cli = GlobalPointerNerLightningCLI(
        GlobalPointerNerModule, GlobalPointerNerDataModule
    )


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="train tplinker ner model")
#     parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

#     parser.add_argument("--train_data", type=str, default="", help="train data path")
#     parser.add_argument("--test_data", type=str, default="", help="test data path")
#     parser.add_argument("--dev_data", type=str, default="", help="dev data path")
#     parser.add_argument("--bert_model", type=str,
#                         default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
#                         help="bert config dir")
#     parser.add_argument("--batch_size", type=int, default=8, help="batch size")
#     parser.add_argument("--max_length", type=int, default=64, help="max input sequence length")
#     parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
#     parser.add_argument("--num_labels", type=int, help="number of entity type")
#     parser.add_argument("--lr_scheduler", choices=["linear", "onecycle", "polydecay"], default="onecycle")
#     parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
#     parser.add_argument("--weight_decay", default=0.01, type=float,
#                         help="Weight decay if we apply some.")
#     parser.add_argument("--warmup_proportion", default=0.1, type=int,
#                         help="warmup steps used for scheduler.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     parser.add_argument("--final_div_factor", type=float, default=1e4,
#                         help="final div factor of linear decay scheduler")

#     parser = pl.Trainer.add_argparse_args(parser)
#     parser = GlobalPointerNerModule.add_model_specific_args(parser)
#     arg = parser.parse_args()
#     model = GlobalPointerNerModule(arg)
#     trainer = pl.Trainer.from_argparse_args(arg, strategy=DDPStrategy(find_unused_parameters=False))
#     datamodule = GlobalPointerNerDataModule(arg)
#     trainer.fit(model, datamodule=datamodule)
