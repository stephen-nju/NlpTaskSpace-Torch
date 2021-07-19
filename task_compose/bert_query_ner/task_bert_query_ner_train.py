# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_bert_mrc.py
@time: 2021/5/18 14:42
"""
import argparse
import os
from typing import Dict, Any, Optional, Union, List
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers.implementations import BertWordPieceTokenizer
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW
from torch.optim import SGD
from datahelper.bert_mrc.mrc_dataset import MRCNERDataset
from datahelper.bert_mrc.mrc_dataset import collate_to_max_length
from metrics.bert_query_ner.query_span_f1 import QuerySpanF1
from modeling.bert_query_ner.modeling_bert_query_ner import BertQueryNER
from modeling.bert_query_ner.configure_bert_query_ner import BertQueryNerConfig
from pytorch_lightning.utilities.seed import seed_everything
from loss.dice_loss import DiceLoss
from pytorch_lightning.callbacks import EarlyStopping

seed_everything(0)


class BertLabeling(pl.LightningModule):
    """MLM Trainer"""

    def __init__(
            self,
            args: argparse.Namespace
    ):
        super().__init__()
        self.args = args
        self.bert_dir = self.args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(self.args.bert_config_dir,
                                                         hidden_dropout_prob=self.args.bert_dropout,
                                                         attention_probs_dropout_prob=self.args.bert_dropout,
                                                         mrc_dropout=self.args.mrc_dropout)

        self.model = BertQueryNER.from_pretrained(self.args.bert_config_dir,
                                                  config=bert_config)
        # logging.info(str(self.args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.loss_type = self.args.loss_type
        if self.loss_type == "bce":
            self.bce_loss = BCEWithLogitsLoss(reduction="none")
        else:
            self.dice_loss = DiceLoss(with_logits=True, smooth=self.args.dice_smooth)
        # todo(yuxian): 由于match loss是n^2的，应该特殊调整一下loss rate
        weight_sum = self.args.weight_start + self.args.weight_end + self.args.weight_span
        self.weight_start = self.args.weight_start / weight_sum
        self.weight_end = self.args.weight_end / weight_sum
        self.weight_span = self.args.weight_span / weight_sum
        self.flat_ner = self.args.flat
        self.span_f1 = QuerySpanF1(compute_on_step=True,
                                   flat=self.flat_ner)
        self.chinese = self.args.chinese
        self.optimizer = self.args.optimizer
        self.span_loss_candidates = self.args.span_loss_candidates
        self.tokenizer = BertWordPieceTokenizer(os.path.join(self.args.bert_config_dir, "vocab.txt"))
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument("--mrc_dropout", type=float, default=0.1,
                                  help="mrc dropout rate")
        model_parser.add_argument("--bert_dropout", type=float, default=0.1,
                                  help="bert dropout rate")
        model_parser.add_argument("--weight_start", type=float, default=1.0)
        model_parser.add_argument("--weight_end", type=float, default=1.0)
        model_parser.add_argument("--weight_span", type=float, default=1.0)
        model_parser.add_argument("--flat", action="store_true", help="is flat ner")
        model_parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "gold"],
                                  default="pred_and_gold", help="Candidates used to compute span loss")
        model_parser.add_argument("--chinese", action="store_true",
                                  help="is chinese dataset")
        model_parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                                  help="loss type")
        model_parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                                  help="loss type")
        model_parser.add_argument("--dice_smooth", type=float, default=1e-8,
                                  help="smooth value of dice loss")
        model_parser.add_argument("--final_div_factor", type=float, default=1e4,
                                  help="final div factor of linear decay scheduler")
        return model_parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon, )
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)

        num_gpus = len([x for x in str(self.trainer.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (
                self.trainer.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps / t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """"""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if self.span_loss_candidates == "all":
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if self.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        if self.loss_type == "bce":
            start_loss = self.bce_loss(start_logits.view(-1), start_labels.view(-1).float())
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = self.bce_loss(end_logits.view(-1), end_labels.view(-1).float())
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            match_loss = self.bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
            match_loss = match_loss * float_match_label_mask
            match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        else:
            start_loss = self.dice_loss(start_logits, start_labels.float(), start_float_label_mask)
            end_loss = self.dice_loss(end_logits, end_labels.float(), end_float_label_mask)
            match_loss = self.dice_loss(span_logits, match_labels.float(), float_match_label_mask)

        return start_loss, end_loss, match_loss

    def training_step(self, batch, batch_idx):

        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = batch
        # num_tasks * [bsz, length, num_labels]
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_loss, end_loss, match_loss = self.compute_loss(start_logits=start_logits,
                                                             end_logits=end_logits,
                                                             span_logits=span_logits,
                                                             start_labels=start_labels,
                                                             end_labels=end_labels,
                                                             match_labels=match_labels,
                                                             start_label_mask=start_label_mask,
                                                             end_label_mask=end_label_mask
                                                             )

        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        # self.log("train_loss", total_loss, prog_bar=True)
        self.log("start_loss", start_loss, prog_bar=True)
        self.log("end_loss", end_loss, prog_bar=True)
        self.log("match_loss", match_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels = batch
        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)
        start_preds, end_preds, match_preds = start_logits > 0, end_logits > 0, span_logits > 0
        self.span_f1(start_preds=start_preds,
                     end_preds=end_preds,
                     match_preds=match_preds,
                     start_label_mask=start_label_mask,
                     end_label_mask=end_label_mask,
                     match_labels=match_labels
                     )

    def on_validation_epoch_end(self) -> None:
        p, r, f1 = self.span_f1.compute()
        print(f"precision {p}", f"recall {r}", f"f1 {f1}", sep="\n", end="\n")
        # self.log("precision", p, prog_bar=True, on_epoch=True)
        # self.log("recall", r, prog_bar=True, on_epoch=True)
        self.log("f1", f1, prog_bar=True, on_epoch=True)

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):

        query, context, tokens, token_type_ids, label_mask = batch

        attention_mask = (tokens != 0).long()
        start_logits, end_logits, span_logits = self(tokens, attention_mask, token_type_ids)

        start_preds, end_preds, match_preds = start_logits > 0, end_logits > 0, span_logits > 0
        # 抽取flat span
        return self.extract_flat_spans(query=query,
                                       context=context,
                                       tokens=tokens,
                                       start_pred=start_preds,
                                       end_pred=end_preds,
                                       match_pred=match_preds,
                                       span_logits=span_logits,
                                       label_mask=label_mask)

    def extract_flat_spans(self, query, context, tokens, start_pred, end_pred, match_pred, span_logits, label_mask):
        batch_entity_span = []
        for query, context, tokens, start_pred, end_pred, match_pred, span_logit, label_mask in zip(query, context,
                                                                                                    tokens,
                                                                                                    start_pred,
                                                                                                    end_pred,
                                                                                                    match_pred,
                                                                                                    span_logits,
                                                                                                    label_mask):
            entity_span = []
            start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
            end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

            for tmp_start in start_positions:
                tmp_ends = [tmp for tmp in end_positions if tmp >= tmp_start]
                for tmp_end in tmp_ends:
                # if len(tmp_end) == 0:
                #     continue
                # else:
                #     tmp_end = min(tmp_end)
                    if match_pred[tmp_start][tmp_end]:
                        prob = span_logit[tmp_start][tmp_end]
                        entity_span.append(self.tokenizer.decode(tokens[tmp_start:tmp_end + 1].tolist()))
            batch_entity_span.append((context, query, entity_span))

        return batch_entity_span

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.get_dataloader(prefix="test")

    def get_dataloader(self, prefix="train", limit: int = None) -> DataLoader:

        json_path = os.path.join(self.data_dir, f"jd_qingdan.topic.corpus.{prefix}.json")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=self.tokenizer,
                                max_length=self.args.max_length,
                                pad_to_maxlen=False
                                )
        # if limit is not None:
        #     dataset = TruncateDataset(dataset, limit)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            # shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length,
        )

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

    parser.add_argument("--data_dir", type=str,
                        default="../data/mrc_extract_data",
                        help="data dir")
    parser.add_argument("--bert_config_dir", type=str,
                        default="E:\\项目资料\\主题挖掘项目\\DataRepository\\bert_model",
                        help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser = BertLabeling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    check_point = ModelCheckpoint(dirpath=args.output_dir)
    early_stop = EarlyStopping("f1", mode="max", patience=3, min_delta=0.2)
    trainer = Trainer.from_argparse_args(parser, callbacks=[check_point, early_stop])

    model = BertLabeling(args)
    trainer.fit(model)
