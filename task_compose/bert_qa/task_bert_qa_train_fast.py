# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_bert_qa.py
@time: 2021/6/9 19:54

fast主要体现在数据处理,tokenizerfast,解码更方便
针对中文数据，采用transformers中的最新方式处理中文qa。使用squad脚本的最大问题就是中文的对齐，早期的
tokenizer代码无法直接解决该问题，所以数据处理脚本非常复杂，好在现在的transformers库(>4.23.1)，
tokenizer中新增了很多字段，能够解决中文的对齐，所以来重构下代码


"""

import argparse
import json
import os
from abc import ABC
from functools import partial
from multiprocessing.dummy import Pool
from os import cpu_count
from typing import Dict, Any, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from transformers import BertTokenizerFast

from datahelper.bert_qa.bert_qa_dataset import QuestionAnswerDataset, QuestionAnswerInputExampleFast, \
    QuestionAnswerInputFeaturesFast
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from metrics.bert_qa.qal_metric import QuestionAnswerMetric, question_answer_evaluation
from modeling.bert_qa.configure_bert_qa import BertForQAConfig
from modeling.bert_qa.modeling_bert_qa import BertForQuestionAnswering

# 设置随机种子
seed_everything(42)

"""
文本处理中的两个问题：1.长文本（需要指定return_overflowing_tokens和stride，用于窗口滑动整个文档），
2：文本对齐（需要指定return_offsets_mapping，用于后处理找到答案位置）
"""


def read_train_data(file):
    # 数据格式发生变化时需要重构的函数
    with open(file, "r", encoding="utf-8") as g:
        s = json.loads(g.read())
        data = s["data"]
        name = s["name"]
        example_index = 0
        for d in data:
            context_text = d["context"]
            for qa in d["qas"]:
                qa_id = qa["id"]
                question = qa["question"]
                answers = qa["answers"]
                is_impossible = False
                if len(answers) == 0:
                    is_impossible = True
                    example = QuestionAnswerInputExampleFast(example_index=example_index,
                                                             title=name,
                                                             question_text=question,
                                                             context_text=context_text,
                                                             is_impossible=True,
                                                             qas_id=qa_id,
                                                             answers=[])
                    example_index += 1
                    yield example
                if not is_impossible:
                    example = QuestionAnswerInputExampleFast(example_index=example_index,
                                                             title=name,
                                                             question_text=question,
                                                             context_text=context_text,
                                                             qas_id=qa_id,
                                                             is_impossible=False,
                                                             answers=answers)
                    example_index += 1
                    yield example


def convert_example_to_features_fast(
        examples: Union[List[QuestionAnswerInputExampleFast], QuestionAnswerInputExampleFast],
        tokenizer,
        max_length,
        doc_stride,
        is_training):
    # 批处理
    features = []
    questions = [example.question_text for example in examples]
    contexts = [example.context_text for example in examples]
    answers_list: List[List[Dict]] = [example.answers for example in examples]
    example_indexs = [example.example_index for example in examples]
    qas_ids = [example.qas_id for example in examples]

    encode_inputs = tokenizer(questions,
                              contexts,
                              max_length=max_length,
                              truncation="only_second",  # 指定改参数，将只在第二部分输入上进行截断，即文章部分进行截断
                              return_overflowing_tokens=True,  # 指定该参数，会根据最大长度与步长将恩本划分为多个段落
                              return_offsets_mapping=True,  # 指定改参数，返回切分后的token在文章中的位置
                              return_token_type_ids=True,
                              return_attention_mask=True,
                              stride=doc_stride,  # 定义重叠token的数目
                              padding="max_length"
                              )

    # sample_mapping中存储着新的片段对应的原始example的id，例如[0, 0, 0, 1, 1, 2]，表示前三个片段都是第1个example
    # 根据sample_mapping中的映射信息，可以有效的定位答案
    sample_mapping = encode_inputs.pop("overflow_to_sample_mapping")
    for i, _ in enumerate(sample_mapping):
        input_ids = encode_inputs["input_ids"][i]
        attention_mask = encode_inputs["attention_mask"][i]
        token_type_ids = encode_inputs["token_type_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = encode_inputs.sequence_ids(i)

        # 问题再前面，文章在后面，拼接起来
        # 定位文章的起始token位置
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # 定位文章的结束token位置
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        offsets = encode_inputs["offset_mapping"][i]

        # 判断答案是否在当前的片段里，条件：文章起始token在原文中的位置要小于答案的起始位置，结束token在原文中的位置要大于答案的结束位置
        # 如果不满足，则将起始与结束位置均置为0
        start_position, end_position = cls_index, cls_index
        is_impossible = False
        answers = answers_list[sample_mapping[i]]  # 根据sample_mapping的结果，获取答案的内容
        if len(answers) == 0:
            is_impossible = True
        else:
            start_char = answers[0]["answer_start"]
            end_char = start_char + len(answers[0]["text"])

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                # 该feature不包含答案
                print("The answer is not in this feature.")
                is_impossible = True
            else:  # 如果满足，则将答案定位到token的位置上
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_position = token_start_index - 1
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_position = token_end_index + 1

        # 定位答案相关
        example_index = example_indexs[sample_mapping[i]]
        # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
        offset_mapping = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(encode_inputs["offset_mapping"][i])
        ]

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
        p_mask = [tok != 1 for tok in encode_inputs.sequence_ids(i)]
        # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
        if tokenizer.cls_token_id is not None:
            cls_indices = np.nonzero(np.array(input_ids) == tokenizer.cls_token_id)[0]
            for cls_index in cls_indices:
                p_mask[cls_index] = 0

        qas_id = qas_ids[i]

        features.append(QuestionAnswerInputFeaturesFast(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        token_type_ids=token_type_ids,
                                                        cls_index=cls_index,
                                                        p_mask=p_mask,
                                                        example_index=example_index,
                                                        start_position=start_position,
                                                        end_position=end_position,
                                                        is_impossible=is_impossible,
                                                        offset_mapping=offset_mapping,
                                                        unique_id=None,
                                                        paragraph_len=0,
                                                        token_is_max_context=0,
                                                        tokens=[],
                                                        qas_id=qas_id,
                                                        encoding=encode_inputs[i]
                                                        ))

    return features


def convert_examples_to_features_pool(examples,
                                      tokenizer,
                                      max_length,
                                      doc_stride,
                                      is_training,
                                      threads=3,
                                      tqdm_enabled=True,
                                      ):
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(convert_example_to_features_fast,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            doc_stride=doc_stride,
                            is_training=is_training,
                            )
        features = list(
                tqdm(
                        p.imap(annotate_, examples, chunksize=32),
                        total=len(examples),
                        desc="convert examples to features",
                        disable=not tqdm_enabled,
                )
        )
        new_features = []
        unique_id = 1000000000
        for example_features in tqdm(
                features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
        ):
            if not example_features:
                continue
            for example_feature in example_features:
                example_feature.unique_id = unique_id
                new_features.append(example_feature)
                unique_id += 1

        features = new_features

        return features


class BertQADataModule(pl.LightningDataModule, ABC):

    def __init__(self,
                 args,
                 train_features,
                 val_examples,
                 val_features
                 ):
        assert isinstance(args, argparse.Namespace)
        self.args = args
        self.train_features = train_features
        self.val_examples = val_examples
        self.val_features = val_features
        super(BertQADataModule, self).__init__()

    def train_dataloader(self):
        return DataLoader(dataset=QuestionAnswerDataset(features=self.train_features),
                          batch_size=self.args.batch_size,
                          num_workers=4,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(dataset=QuestionAnswerDataset(features=self.val_features),
                          batch_size=self.args.batch_size,
                          num_workers=4,
                          pin_memory=True,
                          )


class BertForQA(pl.LightningModule, ABC):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        bert_config = BertForQAConfig.from_pretrained(self.args.bert_config_dir)
        self.model = BertForQuestionAnswering.from_pretrained(self.args.bert_config_dir, config=bert_config)
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_config_dir)
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

    parser.add_argument("--max_query_length",
                        type=int,
                        default=64,
                        help="The maximum number of tokens for the question. "
                             "Questions longer than this will be truncated to this length.")
    # 用于处理
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                             "Sequences longer than this will be truncated, "
                             "and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride",
                        type=int,
                        default=512,
                        help="When splitting up a long document into chunks,"
                             " how much stride to take between chunks.")

    parser.add_argument(
            "--with_negative",
            action="store_true",
            help="If true, the examples contain some that do not have an answer.",
    )
    parser.add_argument("--n_best_size",
                        default=5,
                        type=int,
                        help="The total number of n-best predictions to generate in the nbest_"
                             "predictions.json output file.")
    parser.add_argument(
            "--max_answer_length",
            default=10,
            type=int,
            help="The maximum length of an answer that can be generated."
                 " This is needed because the start "
                 "and end predictions are not conditioned on one another.",
    )

    parser = BertForQA.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    check_point = ModelCheckpoint(dirpath=args.output_dir)
    # early_stop = EarlyStopping("f1", mode="max", patience=3, min_delta=0.2)
    trainer = Trainer.from_argparse_args(parser,
                                         default_root_dir=args.output_dir,
                                         callbacks=[check_point],
                                         strategy=DDPStrategy(find_unused_parameters=False))

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_config_dir)

    train_examples = list(read_train_data(args.train_data))

    train_features = convert_examples_to_features_pool(examples=train_examples,
                                                       tokenizer=tokenizer,
                                                       max_length=args.max_seq_length,
                                                       doc_stride=args.doc_stride,
                                                       is_training=True)
    val_examples = list(read_train_data(args.dev_data))
    val_features = convert_examples_to_features_pool(examples=val_examples,
                                                     tokenizer=tokenizer,
                                                     max_length=args.max_seq_length,
                                                     doc_stride=args.doc_stride,
                                                     is_training=False)

    train_dataloader = DataLoader(dataset=QuestionAnswerDataset(features=train_features),
                                  batch_size=args.batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  )
    val_dataloader = DataLoader(dataset=QuestionAnswerDataset(features=val_features),
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                )

    datamodule = BertQADataModule(args=args, train_features=train_features, val_examples=val_examples,
                                  val_features=val_features)
    model = BertForQA(args)

    trainer.fit(model, datamodule=datamodule)
