# -----coding=utf-8----

import argparse
import os
import pickle

import json

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import BertTokenizerFast

from datahelper.global_pointer import GlobalPointerNerDataset,GlobalPointerNerInputExample,convert_examples_to_features
from modeling.global_pointer.configure_global_pointer import GlobalPointerNerConfig
from modeling.global_pointer.modeling_global_pointer import GlobalPointerNer


seed_everything(42)


class GlobalPointerNerDataModule(pl.LightningDataModule):

    def __init__(self, args) -> None:
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)
        self.label_encode = LabelEncoder()
        self.label_encode.fit(self.get_labels())
        assert self.label_encode.classes_.shape[0]== args.num_labels
        self.mapij2k = {(i, j): trans_ij2k(args.max_length, i, j) for i in range(args.max_length) for j in
                        range(args.max_length) if j >= i}
        self.mapk2ij = {v: k for k, v in self.mapij2k.items()}
        self.cache_path = os.path.join(os.path.dirname(args.train_data), "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        super().__init__()

    def prepare_data(self):
        train_examples = list(self.read_train_data(self.args.train_data))
        train_features = convert_examples_to_features(examples=train_examples,
                                                      tokenizer=self.tokenizer,
                                                      label_encode=self.label_encode,
                                                      max_length=self.args.max_length
                                                      )
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "wb") as g:
            pickle.dump(train_features, g)

        dev_examples = list(self.read_train_data(self.args.dev_data))
        dev_features = convert_examples_to_features(examples=dev_examples,
                                                    tokenizer=self.tokenizer,
                                                    label_encode=self.label_encode,
                                                    max_length=self.args.max_length
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
            for d in data:
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
                yield GlobalPointerNerInputExample(text=context_text,
                                                  labels=labels)

    def get_labels(self):
        return ["HC", "HP","XL"]

    def setup(self, stage: str) -> None:
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "rb") as g:
            self.train_features = pickle.load(g)

        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "rb") as g:
            self.dev_features = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(dataset=GlobalPointerNerDataset(self.train_features),
                          batch_size=self.args.batch_size,
                          num_workers=4,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(dataset=GlobalPointerNerDataset(self.train_features),
                          batch_size=self.args.batch_size,
                          num_workers=4,
                          pin_memory=True
                          )


class GlobalPointerNerModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        config=GlobalPointerNerConfig.from_pretrained(self.args.bert_model)
        config.num_labels = args.num_labels
        self.model = GlobalPointerNer.from_pretrained(args.bert_model, config=config)
        self.loss = MultilabelCategoricalCrossentropy()
        self.optimizer = args.optimizer
        self.metric = NerF1Metric()

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce", help="loss type")
        model_parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="loss type")
        return model_parser

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                        "params"      : [p for n, p in self.model.named_parameters() if
                                         not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                },
                {
                        "params"      : [p for n, p in self.model.named_parameters() if
                                         any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(
                    optimizer_grouped_parameters,
                    betas=(0.9, 0.999),  # according to RoBERTa paper
                    lr=self.args.lr,
                    eps=self.args.adam_epsilon,
            )
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        t_total = (len(self.trainer.datamodule.train_dataloader()) //
                   (self.trainer.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.args.lr,
                                                            pct_start=float(warmup_steps / t_total),
                                                            final_div_factor=self.args.final_div_factor,
                                                            total_steps=t_total,
                                                            anneal_strategy='linear')
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total,
                                                                  lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, inputs, target):
        loss = self.loss(inputs, target)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        label = batch["labels"]
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        total_loss = self.compute_loss(logits, label)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        total_loss = self.compute_loss(logits, labels)
        self.log("valid_loss", total_loss, prog_bar=True)

        mapk2ij = self.trainer.datamodule.mapk2ij
        label_encode = self.trainer.datamodule.label_encode
        self.metric.update(logits, labels, mapk2ij, label_encode)

    def on_validation_end(self):
        p, r, f1 = self.metric.compute()
        print(f"p={p},r={r},f1={f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train tplinker ner model")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

    parser.add_argument("--train_data", type=str, default="", help="train data path")
    parser.add_argument("--test_data", type=str, default="", help="test data path")
    parser.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser.add_argument("--bert_model", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
                        help="bert config dir")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--max_length", type=int, default=64, help="max input sequence length")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--num_labels", type=int, help="number of entity type")
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

    parser = pl.Trainer.add_argparse_args(parser)
    parser = GlobalPointerNerModule.add_model_specific_args(parser)
    arg = parser.parse_args()
    model = GlobalPointerNerModule(arg)
    trainer = pl.Trainer.from_argparse_args(arg, strategy=DDPStrategy(find_unused_parameters=False))
    datamodule = GlobalPointerNerDataModule(arg)
    trainer.fit(model, datamodule=datamodule)
