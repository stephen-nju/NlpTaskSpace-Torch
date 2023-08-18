#------------------- coding:utf-8 -------------

import pickle
import argparse
import json
import os
from abc import ABC
from functools import partial
from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm



from transformers import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from modeling.kbert.modeling_kbert import KBertForQuestionAnswering
from modeling.kbert.configuration_kbert import KBertConfig
from modeling.kbert.tokenization_kbert_fast import KBertTokenizerFast

import lightning as pl
from lightning.pytorch.cli import LightningCLI


class KbertNerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        assert isinstance(args, argparse.Namespace)
        self.args = args
        self.cache_path = os.path.join(os.path.dirname(args.train_data), "cache")
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_config_dir)
        super(KbertNerDataModule, self).__init__()

    def prepare_data(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        train_examples = list(self.read_train_data(self.args.train_data))
        train_features = convert_examples_to_features_pool(
            examples=train_examples,
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length,
            doc_stride=self.args.doc_stride,
            is_training=True,
        )

        val_examples = list(self.read_train_data(self.args.dev_data))
        val_features = convert_examples_to_features_pool(
            examples=val_examples,
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length,
            doc_stride=self.args.doc_stride,
            is_training=False,
        )

        with open(os.path.join(self.cache_path, "train_features.pkl"), "wb") as g:
            pickle.dump(train_features, g)

        with open(os.path.join(self.cache_path, "val_features.pkl"), "wb") as g:
            pickle.dump(val_features, g)

        with open(os.path.join(self.cache_path, "val_examples.pkl"), "wb") as g:
            pickle.dump(val_examples, g)

    @staticmethod
    def read_train_data(file):
        # 数据格式发生变化时需要重构的函数
        with open(file, "r", encoding="utf-8") as g:
            s = json.loads(g.read())
            data = s["data"]
            name = s["name"]
            example_index = 0
            for index, d in enumerate(data):
                # if index>100:break
                context_text = d["context"]
                for qa in d["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    answers = qa["answers"]
                    is_impossible = False
                    if len(answers) == 0:
                        is_impossible = True
                        example = QuestionAnswerInputExampleFast(
                            example_index=example_index,
                            title=name,
                            question_text=question,
                            context_text=context_text,
                            is_impossible=True,
                            qas_id=qa_id,
                            answers=[],
                        )
                        example_index += 1
                        yield example
                    if not is_impossible:
                        example = QuestionAnswerInputExampleFast(
                            example_index=example_index,
                            title=name,
                            question_text=question,
                            context_text=context_text,
                            qas_id=qa_id,
                            is_impossible=False,
                            answers=answers,
                        )
                        example_index += 1
                        yield example

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            with open(os.path.join(self.cache_path, "train_features.pkl"), "rb") as f:
                self.train_features = pickle.load(f)

            with open(os.path.join(self.cache_path, "val_features.pkl"), "rb") as g:
                self.val_features = pickle.load(g)

            with open(os.path.join(self.cache_path, "val_examples.pkl"), "rb") as g:
                self.val_examples = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(
            dataset=QuestionAnswerDataset(features=self.train_features),
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=QuestionAnswerDataset(features=self.val_features),
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class KbertQANerModule(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        bert_config = KBertConfig.from_pretrained(self.args.bert_config_dir)
        self.model = KBertForQuestionAnswering.from_pretrained(
            self.args.bert_config_dir, config=bert_config
        )
        self.tokenizer = KBertTokenizerFast.from_pretrained(args.bert_config_dir)
        self.loss_type = self.args.loss_type
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=self.args.dice_smooth)
        self.optimizer = self.args.optimizer
        self.save_hyperparameters()


    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument(
                "--loss_type", choices=["ce", "bce", "dice"], default="ce", help="loss type"
        )
        model_parser.add_argument(
            "--optimizer", choices=["adamw", "sgd"], default="adamw", help="loss type"
        )
        return model_parser



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
                "weight_decay": self.args.weight_decay,
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
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        t_total = (
            len(self.trainer.datamodule.train_dataloader())
            // (self.trainer.accumulate_grad_batches * num_gpus)
            + 1
        ) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                pct_start=float(warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total,
                anneal_strategy="linear",
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, warmup_steps, t_total, lr_end=self.args.lr / 4.0
            )
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

    def compute_loss(
        self, start_logits, end_logits, start_labels, end_labels, label_mask
    ):
        """compute loss on squad task."""
        if len(start_labels.size()) > 1:
            start_labels = start_labels.squeeze(-1)
        if len(end_labels.size()) > 1:
            end_labels = end_labels.squeeze(-1)

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        batch_size, ignored_index = start_logits.shape  # ignored_index: seq_len
        start_labels.clamp_(0, ignored_index)
        end_labels.clamp_(0, ignored_index)

        if self.loss_type != "ce":
            # start_labels/end_labels: position index of answer starts/ends among the document.
            # F.one_hot will map the postion index to a sequence of 0, 1 labels.
            start_labels = F.one_hot(start_labels, num_classes=ignored_index)
            end_labels = F.one_hot(end_labels, num_classes=ignored_index)

        if self.loss_type == "ce":
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_labels)
            end_loss = loss_fct(end_logits, end_labels)
        elif self.loss_type == "bce":
            start_loss = F.binary_cross_entropy_with_logits(
                start_logits.view(-1), start_labels.view(-1).float(), reduction="none"
            )
            end_loss = F.binary_cross_entropy_with_logits(
                end_logits.view(-1), end_labels.view(-1).float(), reduction="none"
            )

            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()

        elif self.loss_type == "focal":
            # TODO
            # Focal loss
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="none")
            start_loss = loss_fct(
                FocalLoss.convert_binary_pred_to_two_dimension(start_logits.view(-1)),
                start_labels.view(-1),
            )
            end_loss = loss_fct(
                FocalLoss.convert_binary_pred_to_two_dimension(end_logits.view(-1)),
                end_labels.view(-1),
            )
            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()

        elif self.loss_type in ["dice", "adaptive_dice"]:
            loss_fct = DiceLoss(
                with_logits=True,
                smooth=self.args.dice_smooth,
                ohem_ratio=self.args.dice_ohem,
                alpha=self.args.dice_alpha,
                square_denominator=self.args.dice_square,
            )
            # add to test
            # start_logits, end_logits = start_logits.view(batch_size, -1), end_logits.view(batch_size, -1)
            # start_labels, end_labels = start_labels.view(batch_size, -1), end_labels.view(batch_size, -1)
            start_logits, end_logits = start_logits.view(-1, 1), end_logits.view(-1, 1)
            start_labels, end_labels = start_labels.view(-1, 1), end_labels.view(-1, 1)
            # label_mask = label_mask.view(batch_size, -1)
            label_mask = label_mask.view(-1, 1)
            start_loss = loss_fct(start_logits, start_labels, mask=label_mask)
            end_loss = loss_fct(end_logits, end_labels, mask=label_mask)
        else:
            raise ValueError("This type of loss func donot exists.")

        total_loss = (start_loss + end_loss) / 2

        return total_loss, start_loss, end_loss

    def setup(self, stage):
        self.configure_metrics()

    
    def configure_metrics(self):
        all_examples = self.trainer.datamodule.val_examples
        all_features = self.trainer.datamodule.val_features
        evaluation = partial(
            question_answer_evaluation,
            all_examples=all_examples,
            all_features=all_features,
            tokenizer=self.tokenizer,
            n_best_size=5,
            max_answer_length=10,
            do_lower_case=True,
            verbose_logging=True,
            version_2_with_negative=True,
            null_score_diff_threshold=0,
        )
        self.qa_metric = QuestionAnswerMetric(evaluation)


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.global_rank == 0:
            save_path = os.path.join(self.args.output_dir, "saved_model")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_labels = batch["start_position"]
        end_labels = batch["end_position"]
        label_mask = batch["label_mask"]
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)
        total_loss, start_loss, end_loss = self.compute_loss(
            start_logits, end_logits, start_labels, end_labels, label_mask
        )

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("train_start_loss", start_loss, prog_bar=True)
        self.log("train_end_loss", end_loss, prog_bar=True)
        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # 计算验证的loss,验证指标为loss
        # res = []
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        unique_ids = batch["unique_id"]
        start_labels = batch["start_position"]
        end_labels = batch["end_position"]
        label_mask = batch["label_mask"]
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)
        total_loss, start_loss, end_loss = self.compute_loss(
            start_logits, end_logits, start_labels, end_labels, label_mask
        )
        self.log("eval_loss", total_loss, prog_bar=True)
        self.qa_metric.update(unique_ids, start_logits, end_logits)

    def validation_epoch_end(self, outputs) -> None:
        # 使用多GPU训练的时候需要验证
        f1=self.qa_metric.compute()
        self.qa_metric.reset()
        self.log("f1",f1,on_epoch=True)


def main():
    cli=LightningCLI(KbertQANerModule,KbertNerDataModule)


if __name__ == "__main__":
    main()
