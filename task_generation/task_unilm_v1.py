# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: trainer.py
@time: 2021/4/14 14:19
"""
import argparse
import json
import os
import pathlib
import random
from dataclasses import dataclass, asdict
from random import randint, shuffle
from typing import Optional, Any, Union, List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from modeling.unlim_v1.configuration_unilm_v1 import UnilmConfig
from modeling.modeling_unilm import UniLmForGeneration
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from rouge_score import rouge_scorer, scoring
from rouge_score.scoring import AggregateScore, Score
from tokenizers.implementations import BertWordPieceTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from dependence.transformers import get_linear_schedule_with_warmup


@dataclass
class SingleExample:
    input_ids: List[int]
    token_type_ids: List[int]
    src_attention_mask: List[int]
    target_attention_mask: List[int]
    labels: List[int]


@dataclass
class BatchExample:
    batch_src_input_ids: List[List] = None
    batch_target_input_ids: List[List] = None
    batch_src_attention_mask: List[List] = None
    batch_target_attention_mask: List[List] = None
    batch_input_ids: List[List] = None
    batch_token_type_ids: List[List] = None
    batch_attention_mask: List[List] = None
    batch_labels: List[List] = None


class RougeBatchAggregator(scoring.BootstrapAggregator):

    def aggregate(self):
        """
        Override function to wrap the final results in `Score` objects.
        This is due to the scores being replaced with a list of torch tensors.
        """
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple((Score(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def add_scores(self, scores):
        self._scores = scores


class RougeMetric(Metric):
    def __init__(
            self,
            use_stemmer: bool = False,
            rouge_keys: List[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    ):
        super().__init__()
        self.rouge_keys = rouge_keys
        self.use_stemmer = use_stemmer
        self.aggregator = RougeBatchAggregator()
        self.scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=self.use_stemmer)

        for key in rouge_keys:
            self.add_state(key, [])

    def update(self, pred_lns: List[str] = None, tgt_lns: List[str] = None) -> None:
        for pred, tgt in zip(pred_lns, tgt_lns):
            results = self.scorer.score(pred, tgt)
            for key, score in results.items():
                score = torch.tensor([score.precision, score.recall, score.fmeasure])
                getattr(self, key).append(score)

    def compute(self) -> Dict[str, float]:
        scores = {key: getattr(self, key) for key in self.rouge_keys}
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()
        return format_rouge_results(result)

    # def __hash__(self):
    #     # override to hash list objects.
    #     # this is a bug in the upstream pytorch release.
    #     hash_vals = [self.__class__.__name__]
    #
    #     for key in self._defaults.keys():
    #         value = getattr(self, key)
    #         if isinstance(value, list):
    #             value = tuple(value)
    #         hash_vals.append(value)
    #
    #     return hash(tuple(hash_vals))


def format_rouge_results(result: Dict[str, AggregateScore], decimal_places: int = 4) -> Dict[str, float]:
    flattened_result = {}
    for rouge_key, rouge_aggregate_score in result.items():
        for stat in ["precision", "recall", "fmeasure"]:
            mid = rouge_aggregate_score.mid
            score = round(getattr(mid, stat), decimal_places)
            flattened_result[f"{rouge_key}_{stat}"] = score
    return flattened_result


class UnilmDataSetForSeq2Seq(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
        super(UnilmDataSetForSeq2Seq, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        return d


class UnilmDataModule(pl.LightningDataModule):
    def __init__(self, args):
        assert isinstance(args, argparse.Namespace)
        self.args = args
        self.tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_dir, "vocab.txt"))
        self.data_dir = pathlib.Path(args.data_dir)
        self.batch_size = args.batch_size
        self.train = []
        self.test = []
        self.val = []
        super(UnilmDataModule, self).__init__()

    def prepare_data(self):
        pass
        # print("prepare data .....")
        # data = []
        # with open(os.path.join(self.data_dir, "train.txt"), "r", encoding="utf-8") as g:
        #     for line in g:
        #         arr = line.strip().split("\t")
        #         data.append({"src": arr[0], "target": arr[-1]})
        # np.random.shuffle(data)
        # index = int(0.8 * len(data))
        # train = data * 100
        # test = data[index + 1:]
        # with open(os.path.join(self.data_dir, "train.json"), "w", encoding="utf-8") as f:
        #     for a in train:
        #         s = json.dumps(a, ensure_ascii=False)
        #         f.write(s + "\n")
        # print(f"process train data number {len(data)}")
        # with open(os.path.join(self.data_dir, "test.json"), "w", encoding="utf-8") as f:
        #     for a in test:
        #         s = json.dumps(a, ensure_ascii=False)
        #         f.write(s + "\n")

    @staticmethod
    def add_data_specific_args(parent_parse):
        # 添加数据处理时的参数
        data_parser = argparse.ArgumentParser(parents=[parent_parse], add_help=False)
        data_parser.add_argument("--mask_prob", type=float, default=0.15)
        data_parser.add_argument("--max_pred", type=int, default=20)
        # 是否采用ngram mask
        data_parser.add_argument("--skipgram_prb", type=float, default=0)
        data_parser.add_argument("--skipgram_size", type=int, default=1)
        data_parser.add_argument("--max_seq_length", type=int, default=128, help="")
        data_parser.add_argument('--mask_source_words', action='store_true',
                                 help="Whether to mask source words for training")
        data_parser.add_argument('--mask_whole_word', action='store_true',
                                 help="Whether masking a whole word.")
        return data_parser

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_file = self.data_dir.joinpath("train.json")
            with train_file.open(mode="r", encoding="utf-8") as g:
                for line in g:
                    d = json.loads(line.strip())
                    ex = self.train_data_transform(src=d["src"], target=d["target"])
                    self.train.append(ex)
                # self.on_save_checkpoint({"train_data": self.train})
            test_file = self.data_dir.joinpath("test.json")
            with test_file.open(mode="r", encoding="utf-8") as g:
                for line in g:
                    d = json.loads(line.strip())
                    ex = self.train_data_transform(src=d["src"], target=d["target"])
                    self.val.append(ex)

        if stage == "test":
            test_file = self.data_dir.joinpath("test.json")
            with test_file.open(mode="r", encoding="utf-8") as g:
                for line in g:
                    d = json.loads(line.strip())
                    self.test.append(d["src"])

    def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        for k in batch.keys():
            if batch[k] is not None:
                batch[k] = torch.tensor(batch[k], dtype=torch.long).to(device)
        return batch

    def train_data_transform(self, src, target):

        self.tokenizer.enable_truncation(max_length=self.args.max_seq_length)
        if self.tokenizer.padding is not None:
            self.tokenizer.no_padding()
        encode = self.tokenizer.encode(sequence=src, pair=target, add_special_tokens=True)
        tokens = encode.tokens
        labels = encode.ids
        # TODO 是否需要对source 和 target的两部分分别做截断
        type_ids = encode.type_ids
        # segment_ids 使用4和5
        sequence_ids = encode.sequence_ids
        # 计算target的长度
        effective_length = len([i for i in sequence_ids if i is not None and i == 1])
        if self.args.mask_source_words:
            effective_length += len([i for i in sequence_ids if i is not None and i == 0])
        # 计算mask的数量
        n_pred = min(self.args.max_pred, max(1, int(round(effective_length * self.args.mask_prob))))
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        # 还原原始数据的index
        offsets = encode.offsets
        for token_idx in range(len(tokens)):
            token_start, token_end = offsets[token_idx]

            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        cand_pos = []
        special_pos = set()
        # 构建mask 利用type_ids 判是否是target
        for index, (tp, tok) in enumerate(zip(type_ids, tokens)):
            if self.args.mask_source_words:
                if tp == 0:
                    if tok not in ["[CLS]", "[SEP"]:
                        cand_pos.append(index)
                    else:
                        special_pos.add(index)
                else:
                    cand_pos.append(index)
            else:
                if tp == 1:
                    cand_pos.append(index)
                else:
                    special_pos.add(index)

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            if (self.args.skipgram_prb > 0) and (self.args.skipgram_size >= 2) and (
                    random.random() < self.args.skipgram_prb):
                # ngram mask
                cur_skipgram_size = randint(2, self.args.skipgram_size)
                if self.args.mask_whole_word:
                    st_pos, end_pos = origin_offset2token_idx_start[pos], origin_offset2token_idx_end[
                        pos + cur_skipgram_size]
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if self.args.mask_whole_word:
                    st_pos, end_pos = origin_offset2token_idx_start[pos], origin_offset2token_idx_end[pos + 1]
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        # 可能会多选
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        for pos in masked_pos:
            if random.random() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif random.random() < 0.5:  # 10%
                tokens[pos] = self.tokenizer.id_to_token(random.randint(0, self.tokenizer.get_vocab_size() - 1))
        input_ids = [self.tokenizer.token_to_id(token) for token in tokens]
        for pos, ids in enumerate(labels):
            if pos not in masked_pos:
                labels[pos] = -100

        attention_mask = encode.attention_mask
        src_attention_mask = []
        target_attention_mask = []
        for index, v in enumerate(attention_mask):
            if type_ids[index] == 0:
                src_attention_mask.append(attention_mask[index])
            else:
                target_attention_mask.append(attention_mask[index])

        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]
        """
        return SingleExample(input_ids=input_ids,
                             token_type_ids=type_ids,
                             src_attention_mask=src_attention_mask,
                             target_attention_mask=target_attention_mask,
                             labels=labels)

    def train_collate_fn(self, batch: List[SingleExample]):
        # padding batch
        max_length = max(len(x.input_ids) for x in batch)
        padding = self.tokenizer.token_to_id("[PAD]")
        batch_inputs_ids = [x.input_ids for x in batch]
        batch_token_type_ids = [x.token_type_ids for x in batch]
        batch_labels = [x.labels for x in batch]
        input_ids = self.padding_sequence(np.asarray(batch_inputs_ids), max_length, padding)
        token_type_ids = self.padding_sequence(np.asarray(batch_token_type_ids), max_length, padding)
        # 构建attention mask
        batch_attention_mask = []
        # TODO 用于构建验证集步骤时候的数据
        # batch_src_attention_mask = []
        # batch_target_attention_mask=[]
        # batch_src_input_ids=[]
        # batch_target_input_ids=[]
        for x in batch:
            # 构建attention mask的矩阵
            padding_attention_mask = np.zeros((max_length, max_length))
            source_attention_mask = np.asarray(x.src_attention_mask)
            target_attention_mask = np.asarray(x.target_attention_mask)
            seq_len = source_attention_mask.shape[0] + target_attention_mask.shape[0]
            # 扩充到seq len
            new_source_attention_mask = np.pad(source_attention_mask,
                                               pad_width=(0, target_attention_mask.shape[0]),
                                               mode="constant", constant_values=(0, 0))

            new_target_attention_mask = np.pad(target_attention_mask,
                                               pad_width=(source_attention_mask.shape[0], 0),
                                               mode="constant", constant_values=(0, 0))
            source = np.tile(new_source_attention_mask, (seq_len, 1))
            target = np.tile(new_target_attention_mask, (seq_len, 1))
            attention_mask = source + np.tril(target)
            padding_attention_mask[:seq_len, :seq_len] = attention_mask
            batch_attention_mask.append(padding_attention_mask)

        # 构建attention mask的新方式
        """
        a=np.asarray([0,0,0,1,1])
        b=np.tile(a,(5,1))
        idx=np.cumsum(b,axis=1)
        mask=idx[:,None,:]<=idx[:,:,None]
        mask=mask[:,None]
        """
        attention_mask = np.asarray(batch_attention_mask)
        labels = self.padding_sequence(batch_labels, max_length, padding=-100)

        batch_example = BatchExample(
            batch_input_ids=input_ids,
            batch_token_type_ids=token_type_ids,
            batch_labels=labels,
            batch_attention_mask=attention_mask,
        )
        return asdict(batch_example)

    def predict_collate_fn(self, batch: List):

        if self.tokenizer.padding is None:
            self.tokenizer.enable_padding()

        encode = self.tokenizer.encode_batch(batch)
        inputs_ids = [e.ids for e in encode]
        # tokens = [e.tokens for e in encode]
        type_ids = [e.type_ids for e in encode]
        # output["input_text"] = tokens
        batch_src_attention_mask = [e.attention_mask for e in encode]
        input_ids = torch.tensor(inputs_ids, dtype=torch.long)
        token_type_ids = torch.tensor(type_ids, dtype=torch.long)

        batch_example = BatchExample(batch_input_ids=input_ids,
                                     batch_token_type_ids=token_type_ids,
                                     batch_src_attention_mask=batch_src_attention_mask
                                     )
        return asdict(batch_example)

    @staticmethod
    def padding_sequence(inputs, length, padding):
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        outputs = []
        for x in inputs:
            x = x[:length]
            pad_width[0] = (0, length - len(x))
            x = np.pad(x, pad_width, 'constant', constant_values=padding)
            outputs.append(x)

        return np.array(outputs)

    def train_dataloader(self) -> Any:
        return DataLoader(dataset=UnilmDataSetForSeq2Seq(data=self.train),
                          batch_size=self.batch_size,
                          # num_workers=4,
                          pin_memory=True,
                          collate_fn=self.train_collate_fn
                          )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=UnilmDataSetForSeq2Seq(data=self.val),
                          batch_size=self.batch_size,
                          # num_workers=4,
                          pin_memory=True,
                          collate_fn=self.train_collate_fn
                          )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=UnilmDataSetForSeq2Seq(self.test), batch_size=self.batch_size,
                          collate_fn=self.train_collate_fn)

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=UnilmDataSetForSeq2Seq(data=self.test),
                          batch_size=self.batch_size,
                          collate_fn=self.predict_collate_fn)


class UniLmLightModule(pl.LightningModule):
    def __init__(self,
                 args
                 ):
        assert isinstance(args, argparse.Namespace)
        # 保证传参的类型
        super(UniLmLightModule, self).__init__()
        self.args = args
        self.tokenizer = BertWordPieceTokenizer(os.path.join(args.bert_dir, "vocab.txt"))
        self.bert_model = pathlib.Path(args.bert_dir)
        self.config = UnilmConfig.from_pretrained(args.bert_dir)
        self.model = UniLmForGeneration.from_pretrained(self.bert_model, config=self.config)
        self.rouge = RougeMetric(use_stemmer=False)  # 词干分析
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parse):
        # 添加模型级别的参数
        model_parse = argparse.ArgumentParser(parents=[parent_parse], add_help=False)
        model_parse.add_argument("--optimizer", choices=["adamw", "adam"], help="")
        model_parse.add_argument("--lr", type=float, default=5e-5, help="learning rate")
        model_parse.add_argument("--warmup_proportion", default=0.1, type=float,
                                 help="Proportion of training to perform linear learning rate warmup for. "
                                      "E.g., 0.1 = 10%% of training.")
        model_parse.add_argument("--adam_epsilon", default=1e-8, type=float,
                                 help="Epsilon for Adam optimizer.")
        return model_parse

    def training_step(self, batch, batch_idx):
        inputs_ids = batch["batch_input_ids"]
        type_ids = batch["batch_token_type_ids"]
        attention_mask = batch["batch_attention_mask"]
        labels = batch["batch_labels"]

        out = self.model(input_ids=inputs_ids,
                         token_type_ids=type_ids,
                         attention_mask=attention_mask,
                         labels=labels,
                         return_dict=True
                         )
        return out.loss

    def validation_step(self, batch, batch_idx, **kwargs):
        inputs_ids = batch["batch_input_ids"]
        type_ids = batch["batch_token_type_ids"]
        attention_mask = batch["batch_attention_mask"]
        labels = batch["batch_labels"]

        out = self.model(input_ids=inputs_ids,
                         token_type_ids=type_ids,
                         attention_mask=attention_mask,
                         labels=labels,
                         return_dict=True
                         )

        self.log(name="val_loss", value=out.loss, prog_bar=True)
        # generate = self.model.generate()

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        inputs_ids = batch["batch_input_ids"]
        type_ids = batch["batch_token_type_ids"]
        src_attention_mask = batch["batch_src_attention_mask"]
        labels = batch["batch_labels"]

        out = self.model.generate(input_ids=inputs_ids,
                                  attention_mask=src_attention_mask,
                                  token_type_ids=type_ids,
                                  pad_token_id=self.tokenizer.token_to_id("[PAD]"),
                                  eos_token_id=self.tokenizer.token_to_id("[SEP]"),
                                  mask_token_ids=self.tokenizer.token_to_id("[MASK]"))
        for sentence in out:
            print(self.tokenizer.decode(sentence.tolist(), skip_special_tokens=False))
        return out

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        num_gpus = len([x for x in str(self.trainer.gpus).split(",") if x.strip()])

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_epsilon)
        t_total = (len(self.train_dataloader()) // (
                self.trainer.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(self.args.warmup_proportion * t_total),
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_callbacks(self):
        check_point = ModelCheckpoint()
        early_stop = EarlyStopping(monitor="val_loss")
        return [check_point, early_stop]


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Training")
    parse.add_argument("--data_dir", type=str,
                       default="/home/nlpbigdata/net_disk_project/zhubin/gds_associate_project/data_repository/unilm_generation_data/test_data/",
                       help="")
    parse.add_argument("--bert_dir", type=str,
                       default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
                       help="")
    parse.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parse.add_argument("--batch_size", type=int, default=8, help="")

    parse = Trainer.add_argparse_args(parse)
    parse = UniLmLightModule.add_model_specific_args(parse)
    parse = UnilmDataModule.add_data_specific_args(parse)
    parse = UnilmDataModule.add_argparse_args(parse)
    arg = parse.parse_args()

    trainer = Trainer.from_argparse_args(arg,
                                         num_sanity_val_steps=-1,
                                         plugins=DDPPlugin(find_unused_parameters=False))
    model = UniLmLightModule(arg)
    data = UnilmDataModule(arg)
    trainer.fit(model=model, datamodule=data)
