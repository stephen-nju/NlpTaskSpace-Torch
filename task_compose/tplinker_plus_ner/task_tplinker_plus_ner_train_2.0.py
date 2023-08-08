# -----*----coding:utf8-----*----
import argparse
import json
import os
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import BertTokenizerFast

from datahelper.tplinker_plus.tplinker_plus_dataset import (
    convert_examples_to_features,
    TplinkerPlusNerDataset,
    TplinkerPlusNerInputExample,
    trans_ij2k,
)
from metrics.tplinker_plus_ner.ner_f1 import NerF1Metric
from modeling.tplinker_plus.configure_tplinker_plus import TplinkerPlusNerConfig
from modeling.tplinker_plus.modeling_tplinker_plus import TplinkerPlusNer
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_weights = None

    def GHM(self, gradient, bins=10, beta=0.9):
        """
        gradient_norm: gradient_norms of all examples in this batch; (batch_size, shaking_seq_len)
        """
        avg = torch.mean(gradient)
        std = torch.std(gradient) + 1e-12
        gradient_norm = torch.sigmoid(
            (gradient - avg) / std
        )  # normalization and pass through sigmoid to 0 ~ 1.

        min_, max_ = torch.min(gradient_norm), torch.max(gradient_norm)
        gradient_norm = (gradient_norm - min_) / (max_ - min_)
        gradient_norm = torch.clamp(
            gradient_norm, 0, 0.9999999
        )  # ensure elements in gradient_norm != 1.

        example_sum = torch.flatten(gradient_norm).size()[0]  # N

        # calculate weights
        current_weights = torch.zeros(bins).to(gradient.device)
        hits_vec = torch.zeros(bins).to(gradient.device)
        count_hits = 0  # coungradient_normof hits
        for i in range(bins):
            bar = float((i + 1) / bins)
            hits = torch.sum((gradient_norm <= bar)) - count_hits
            count_hits += hits
            hits_vec[i] = hits.item()
            current_weights[i] = example_sum / bins / (hits.item() + example_sum / bins)
        # EMA: exponential moving averaging
        #         print()
        #         print("hits_vec: {}".format(hits_vec))
        #         print("current_weights: {}".format(current_weights))
        if self.last_weights is None:
            self.last_weights = torch.ones(bins).to(gradient.device)  # init by ones
        current_weights = self.last_weights * beta + (1 - beta) * current_weights
        self.last_weights = current_weights
        #         print("ema current_weights: {}".format(current_weights))

        # weights4examples: pick weights for all examples
        weight_pk_idx = (gradient_norm / (1 / bins)).long()[:, :, None]
        weights_rp = current_weights[None, None, :].repeat(
            gradient_norm.size()[0], gradient_norm.size()[1], 1
        )
        weights4examples = torch.gather(weights_rp, -1, weight_pk_idx).squeeze(-1)
        weights4examples /= torch.sum(weights4examples)
        return weights4examples * gradient  # return weighted gradients

    def forward(self, y_pred, y_true):
        """y_true ([Tensor]): [..., num_classes]
        y_pred ([Tensor]): [..., num_classes]
        """

        # print(f"y_pred=={y_pred.requires_grad}")
        # print(f"y_true=={y_true.requires_grad}")
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12

        y_pred_pos = torch.cat(
            [y_pred_pos, torch.zeros_like(y_pred_pos[..., :1],requires_grad=True)], dim=-1
        )
        y_pred_neg = torch.cat(
            [y_pred_neg, torch.zeros_like(y_pred_neg[..., :1],requires_grad=True)], dim=-1
        )
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (self.GHM(neg_loss + pos_loss, bins=1000)).sum()
        # return (pos_loss + neg_loss).mean()


class TplinkerPlusNerDataModule(pl.LightningDataModule):
    def __init__(
        self, train_data, dev_data, max_length, batch_size, bert_model, num_labels,workers
    ) -> None:
    
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.label_encode = LabelEncoder()
        self.label_encode.fit(self.get_labels())
        assert self.label_encode.classes_.shape[0] == num_labels
        self.mapij2k = {
            (i, j): trans_ij2k(max_length, i, j)
            for i in range(max_length)
            for j in range(max_length)
            if j >= i
        }
        self.mapk2ij = {v: k for k, v in self.mapij2k.items()}
        self.cache_path = os.path.join(os.path.dirname(train_data), "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.train_data = train_data
        self.dev_data = dev_data
        self.max_length = max_length
        self.batch_size = batch_size
        self.workers=workers
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
                yield TplinkerPlusNerInputExample(text=context_text, labels=labels)

    def get_labels(self):
        return ["HC", "HP", "XL"]

    def setup(self, stage: str) -> None:
        with open(os.path.join(self.cache_path, "train_feature.pkl"), "rb") as g:
            self.train_features = pickle.load(g)

        with open(os.path.join(self.cache_path, "dev_feature.pkl"), "rb") as g:
            self.dev_features = pickle.load(g)

    def train_dataloader(self):
        return DataLoader(
            dataset=TplinkerPlusNerDataset(self.train_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=TplinkerPlusNerDataset(self.dev_features),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class TplinkerPlusNerModule(pl.LightningModule):
    def __init__(
        self,
        bert_model,
        num_labels,
        loss_type,
        optimizer,
        lr,
        handshaking_lr,
        lr_scheduler,
        weight_decay,
        adam_epsilon,
        warmup_proportion,
        final_div_factor,
        rewarm_epoch_num,
    ):
        super().__init__()
        
        self.num_labels=num_labels
        self.lr_scheduler=lr_scheduler
        self.lr = lr
        self.handshaking_lr = handshaking_lr
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.warmup_proportion = warmup_proportion
        self.rewarm_epoch_num = rewarm_epoch_num
        self.final_div_factor = final_div_factor

        config = TplinkerPlusNerConfig.from_pretrained(bert_model)
        config.num_labels = num_labels
        self.model = TplinkerPlusNer.from_pretrained(bert_model, config=config)
        self.loss = MultilabelCategoricalCrossentropy()
        self.optimizer = optimizer
        self.metric = NerF1Metric()
        self.save_hyperparameters()

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler"""
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param = []
        handshaking_param = []
        for name, param in self.model.named_parameters():
            # print(f"{name}=={param.requires_grad}")
            space = name.split(".")
            if space[0] == "bert":
                bert_param.append((name, param))
            else:
                handshaking_param.append((name, param))

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in bert_param if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.lr,
            },
            {
                "params": [
                    p for n, p in bert_param if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.lr,
            },
            {
                "params": [
                    p for n, p in handshaking_param if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
                "lr": self.handshaking_lr,
            },
            {
                "params": [
                    p for n, p in handshaking_param if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.handshaking_lr,
            },
        ]

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.lr
            )
        elif self.optimizer=="adamw":
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.lr,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.lr,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay)
        # num_gpus = len([x for x in str(self.trainer.gpus).split(",") if x.strip()])
        num_gpus=self.trainer.num_devices
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
            # TODO
            step = (len(self.trainer.datamodule.train_dataloader())) // (
                self.trainer.accumulate_grad_batches * num_gpus + 1
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, step * self.rewarm_epoch_num, 1
            )
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids, is_training=True):
        """forward inputs to BERT models."""
        return self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            is_training=is_training,
        )

    def compute_loss(self, inputs, target):
        loss = self.loss.forward(inputs, target)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        label = batch["labels"]
        # logits=self.forward(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,is_training=False)
        logits, sampled_tok_pair_indices = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            is_training=True
        )
        label = label.gather(
            1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, self.num_labels)
        )
        total_loss = self.compute_loss(logits, label)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

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
            is_training=False,
        )
        total_loss = self.compute_loss(logits, labels)
        self.log("valid_loss", total_loss, prog_bar=True)

        mapk2ij = self.trainer.datamodule.mapk2ij
        label_encode = self.trainer.datamodule.label_encode
        self.metric.update(logits, labels, mapk2ij, label_encode)

    def on_validation_end(self):
        p, r, f1 = self.metric.compute()
        print(f"p={p},r={r},f1={f1}")


class TplinkerPlusNerLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.bert_model", "data.bert_model")
        parser.link_arguments("model.num_labels", "data.num_labels")
        # parser.link_arguments("data.batch_size","model.batch_size")


def main():
    cli = TplinkerPlusNerLightningCLI(TplinkerPlusNerModule, TplinkerPlusNerDataModule)


if __name__ == "__main__":
    main()
