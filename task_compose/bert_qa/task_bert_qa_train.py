# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_bert_qa.py
@time: 2021/6/9 19:54
"""

import argparse
import json
import os
from abc import ABC
from functools import partial
from typing import Optional, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BertTokenizer

from datahelper.bert_qa.bert_qa_dataset import QuestionAnswerDataset, convert_examples_to_features, QuestionAnswerInputExample
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from metrics.bert_qa.qal_metric import QuestionAnswerMetric, question_answer_evaluation
from modeling.bert_qa.configure_bert_qa import BertForQAConfig
from modeling.bert_qa.modeling_bert_qa import BertForQuestionAnswering
from pytorch_lightning.strategies import DDPStrategy

# 设置随机种子
seed_everything(42)


class BerQADataModule(pl.LightningDataModule, ABC):

    def __init__(self, args):
        assert isinstance(args, argparse.Namespace)
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_config_dir)
        self.train_data = args.train_data
        self.test_data = args.test_data
        self.dev_data = args.dev_data
        self.batch_size = args.batch_size
        self.train_feature = []
        self.val_examples = []
        self.val_features = []
        super(BerQADataModule, self).__init__()

    @staticmethod
    def read_train_data(file):
        # 数据格式发生变化时需要重构的函数
        with open(file, "r", encoding="utf-8") as g:
            s = json.loads(g.read())
            data = s["data"]
            name = s["name"]
            for d in data:
                context_text = d["context"]
                for qa in d["qas"]:
                    qa_id = qa["id"]
                    question = qa["question"]
                    answers = qa["answers"]
                    is_impossible = False
                    if len(answers) == 0:
                        is_impossible = True
                        example = QuestionAnswerInputExample(qas_id=qa_id,
                                                             title=name,
                                                             question_text=question,
                                                             context_text=context_text,
                                                             answer_text=None,
                                                             raw_start_position=None,
                                                             is_impossible=True,
                                                             answers=[])
                        yield example
                    if not is_impossible:
                        for ans in answers:
                            example = QuestionAnswerInputExample(qas_id=qa_id,
                                                                 title=name,
                                                                 question_text=question,
                                                                 context_text=context_text,
                                                                 answer_text=ans["text"],
                                                                 raw_start_position=ans["answer_start"],
                                                                 is_impossible=False,
                                                                 answers=[ans])
                            yield example
        # with open(file, "r", encoding="utf-8") as g:
        #     index = 0
        #     # 构建qa对的时候，设置唯一的id,qas_id
        #     for line in g:
        #         lines = line.strip().split("\t")
        #         assert len(lines) == 2
        #         title = None
        #         question_text = "标题中产品提及有哪些"
        #         context_text = lines[0]
        #         ths = json.loads(lines[1])
        #         is_impossible = False
        #         if len(ths) == 0:
        #             is_impossible = True
        #             example = QAInputExample(qas_id=index,
        #                                      title=title,
        #                                      question_text=question_text,
        #                                      context_text=context_text,
        #                                      answer_text=None,
        #                                      raw_start_position=None,
        #                                      is_impossible=True,
        #                                      answers=[])
        #             index += 1
        #             yield example
        #         if not is_impossible:
        #             for tx in ths:
        #                 data = re.finditer(tx, context_text)
        #                 start_index = []
        #                 for d in data:
        #                     start_index.append(d.start(0))
        #                 example = QAInputExample(qas_id=index,
        #                                          title=title,
        #                                          question_text=question_text,
        #                                          context_text=context_text,
        #                                          answer_text=tx,
        #                                          raw_start_position=start_index[0],
        #                                          is_impossible=False,
        #                                          answers=[])
        #                 index += 1
        #                 yield example

    @staticmethod
    def add_data_specific_args(parent_parse):
        # 添加数据处理时的参数
        data_parser = argparse.ArgumentParser(parents=[parent_parse], add_help=False)
        data_parser.add_argument("--max_query_length",
                                 type=int,
                                 default=64,
                                 help="The maximum number of tokens for the question. "
                                      "Questions longer than this will be truncated to this length.")
        # 用于处理
        data_parser.add_argument("--max_seq_length",
                                 type=int,
                                 default=384,
                                 help="The maximum total input sequence length after WordPiece tokenization. "
                                      "Sequences longer than this will be truncated, "
                                      "and sequences shorter than this will be padded.")
        data_parser.add_argument("--doc_stride",
                                 type=int,
                                 default=512,
                                 help="When splitting up a long document into chunks,"
                                      " how much stride to take between chunks.")

        data_parser.add_argument(
            "--with_negative",
            action="store_true",
            help="If true, the examples contain some that do not have an answer.",
        )
        data_parser.add_argument("--n_best_size",
                                 default=20,
                                 type=int,
                                 help="The total number of n-best predictions to generate in the nbest_"
                                      "predictions.json output file.")
        data_parser.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help="The maximum length of an answer that can be generated."
                 " This is needed because the start "
                 "and end predictions are not conditioned on one another.",
        )
        return data_parser

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_file = self.train_data
            self.train_feature = convert_examples_to_features(examples=list(self.read_train_data(train_file)),
                                                              tokenizer=self.tokenizer,
                                                              max_query_length=self.args.max_query_length,
                                                              max_seq_length=self.args.max_seq_length,
                                                              doc_stride=self.args.doc_stride,
                                                              is_training=True)
            self.val_examples = list(self.read_train_data(train_file))
            self.val_features = convert_examples_to_features(examples=self.val_examples,
                                                             tokenizer=self.tokenizer,
                                                             max_query_length=self.args.max_query_length,
                                                             max_seq_length=self.args.max_seq_length,
                                                             doc_stride=self.args.doc_stride,
                                                             is_training=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=QuestionAnswerDataset(features=self.train_feature),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=QuestionAnswerDataset(features=self.val_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class BertForQA(pl.LightningModule, ABC):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        bert_config = BertForQAConfig.from_pretrained(self.args.bert_config_dir)
        self.model = BertForQuestionAnswering.from_pretrained(self.args.bert_config_dir, config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_config_dir)
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
        model_parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce", help="loss type")
        model_parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="loss type")
        return model_parser

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
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

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels, label_mask):
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
            start_loss = F.binary_cross_entropy_with_logits(start_logits.view(-1),
                                                            start_labels.view(-1).float(),
                                                            reduction="none")
            end_loss = F.binary_cross_entropy_with_logits(end_logits.view(-1), end_labels.view(-1).float(),
                                                          reduction="none")

            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()
        elif self.loss_type == "focal":
            # TODO
            # Focal loss
            loss_fct = FocalLoss(gamma=self.args.focal_gamma, reduction="none")
            start_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(start_logits.view(-1)),
                                  start_labels.view(-1))
            end_loss = loss_fct(FocalLoss.convert_binary_pred_to_two_dimension(end_logits.view(-1)),
                                end_labels.view(-1))
            start_loss = (start_loss * label_mask.view(-1)).sum() / label_mask.sum()
            end_loss = (end_loss * label_mask.view(-1)).sum() / label_mask.sum()

        elif self.loss_type in ["dice", "adaptive_dice"]:
            loss_fct = DiceLoss(with_logits=True,
                                smooth=self.args.dice_smooth,
                                ohem_ratio=self.args.dice_ohem,
                                alpha=self.args.dice_alpha,
                                square_denominator=self.args.dice_square)
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
        evaluation = partial(question_answer_evaluation,
                             all_examples=all_examples,
                             all_features=all_features,
                             tokenizer=self.tokenizer,
                             n_best_size=5,
                             max_answer_length=10,
                             do_lower_case=True,
                             verbose_logging=True,
                             version_2_with_negative=True,
                             null_score_diff_threshold=0)
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
        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,
                                                             label_mask)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("train_start_loss", start_loss, prog_bar=True)
        self.log("train_end_loss", end_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

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
        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,
                                                             label_mask)
        self.log("eval_total_loss", total_loss, prog_bar=True)
        self.qa_metric.update(unique_ids, start_logits, end_logits)

    def validation_epoch_end(self, outputs) -> None:
        # 使用多GPU训练的时候需要验证
        self.qa_metric.compute()
        self.qa_metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

    parser.add_argument("--train_data", type=str, default="", help="train data path")
    parser.add_argument("--test_data", type=str, default="", help="test data path")
    parser.add_argument("--dev_data", type=str, default="", help="dev data path")
    parser.add_argument("--bert_config_dir", type=str, default="", help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", choices=["linear", "onecycle", "polydecay"], default="onecycle")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")
    ## dice loss
    parser.add_argument("--dice_smooth", type=float, default=1e-4, help="smooth value of dice loss")
    parser.add_argument("--dice_ohem", type=float, default=0.0, help="ohem ratio of dice loss")
    parser.add_argument("--dice_alpha", type=float, default=0.01, help="alpha value of adaptive dice loss")
    parser.add_argument("--dice_square", action="store_true", help="use square for dice loss")
    ## focal loss
    parser.add_argument("--focal_gamma", type=float, default=2, help="gamma for focal loss.")
    parser.add_argument("--focal_alpha", type=float, help="alpha for focal loss.")

    parser = BertForQA.add_model_specific_args(parser)
    parser = BerQADataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    check_point = ModelCheckpoint(dirpath=args.output_dir)
    # early_stop = EarlyStopping("f1", mode="max", patience=3, min_delta=0.2)
    trainer = Trainer.from_argparse_args(parser,
                                         default_root_dir=args.output_dir,
                                         callbacks=[check_point],
                                         strategy=DDPStrategy(find_unused_parameters=False))
    data_module = BerQADataModule(args)
    model = BertForQA(args)
    trainer.fit(model, datamodule=data_module)
