# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: task_tplinker_joint_extract.py
@time: 2021/5/13 14:52
"""
# 片段中实体关系数量较多，模型效果越突出
import sys
import argparse
import json
import os
import pathlib
from typing import Optional
import torch
import unicodedata
import pytorch_lightning as pl
from torch import nn
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup, AdamW
from modeling.tplinker.modeling_tplinker_pytorch import TPLinkerBert
from datahelper.tplinker.preprocess import Preprocessor
from datahelper.tplinker.tplinker_dataset import DataMaker4Bert, HandshakingTaggingScheme, TplinkerDataset


class TplinkerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(TplinkerDataModule, self).__init__()
        assert isinstance(args, argparse.Namespace)
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
        self.data_dir = pathlib.Path(args.data_dir)
        self.cache_dir = self.data_dir.joinpath("cache")
        self.train_feature = []
        # self.get_tok2char_span_map = self.get_span_map
        self.preprocessor = Preprocessor(tokenize_func=self.tokenizer.tokenize,
                                         get_tok2char_span_map_func=self.get_tok2char_span_map
                                         )
        self.batch_size = args.batch_size
        self.relation_size = self.compute_relation_size()

    def compute_relation_size(self):
        rel = json.load(open(os.path.join(self.cache_dir, "rel2id.json"), "r"))
        return len(rel)

    def get_tok2char_span_map(self, text):
        return self.tokenizer.encode_plus(text,
                                          return_offsets_mapping=True,
                                          add_special_tokens=False)["offset_mapping"]

    @staticmethod
    def add_data_specific_args(parent_parse):
        # 添加数据处理时的参数
        data_parser = argparse.ArgumentParser(parents=[parent_parse], add_help=False)
        # 用于处理
        data_parser.add_argument("--max_seq_length", type=int, default=128,
                                 help="The maximum total input sequence length after WordPiece tokenization. "
                                      "Sequences longer than this will be truncated, "
                                      "and sequences shorter than this will be padded.")

        return data_parser

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = json.load(open(os.path.join(self.cache_dir, "train.json"), "r", encoding="utf-8"), )
            max_tok_num = 0
            for sample in data:
                tokens = self.tokenizer.tokenize(sample["text"])
                max_tok_num = max(max_tok_num, len(tokens))
            max_seq_len = min(max_tok_num, self.args.max_seq_length)
            rel2id = json.load(open(os.path.join(self.cache_dir, "rel2id.json"), "r", encoding="utf-8"))
            self.handshaking_tagger = HandshakingTaggingScheme(relation2id=rel2id, max_seq_len=max_seq_len)
            self.data_maker = DataMaker4Bert(tokenizer=self.tokenizer,
                                             handshaking_tagger=self.handshaking_tagger
                                             )
            data = self.preprocessor.split_into_short_samples(sample_list=data,
                                                              max_seq_len=max_seq_len,
                                                              sliding_len=50,
                                                              encoder="BERT",
                                                              data_type="train"
                                                              )
            self.train_feature = self.get_indexed_data(data, max_seq_len=max_seq_len)

    def get_indexed_data(self, data, max_seq_len, data_type="train"):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc="Generate indexed train or valid data"):
            text = sample["text"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text,
                                               return_offsets_mapping=True,
                                               add_special_tokens=False,
                                               max_length=max_seq_len,
                                               truncation=True,
                                               pad_to_max_length=True)

            # tagging
            spots_tuple = None
            if data_type != "test":
                spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (sample,
                         input_ids,
                         attention_mask,
                         token_type_ids,
                         tok2char_span,
                         spots_tuple,
                         )
            indexed_samples.append(sample_tp)
        return indexed_samples

    def generate_batch(self, batch_data, data_type="train"):
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        tok2char_span_list = []

        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for tp in batch_data:
            sample_list.append(tp[0])
            input_ids_list.append(tp[1])
            attention_mask_list.append(tp[2])
            token_type_ids_list.append(tp[3])
            tok2char_span_list.append(tp[4])

            if data_type != "test":
                ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = tp[5]
                ent_spots_list.append(ent_matrix_spots)
                head_rel_spots_list.append(head_rel_matrix_spots)
                tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = None, None, None
        if data_type != "test":
            batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
            batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
            batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return sample_list, \
               batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
               batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag

    def train_dataloader(self):
        return DataLoader(dataset=TplinkerDataset(features=self.train_feature),
                          batch_size=self.batch_size,
                          num_workers=4,
                          collate_fn=self.generate_batch,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(dataset="",
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=True,
                          )


class TplinkerExtraction(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, relation_size: int):
        super(TplinkerExtraction, self).__init__()
        self.args = args
        self.relation_size = relation_size
        self.tokenizer = BertTokenizerFast.from_pretrained(self.args.bert_dir)
        self.encoder = BertModel.from_pretrained(self.args.bert_dir)
        self.optimizer = self.args.optimizer
        self.tplinker_model = TPLinkerBert(encoder=self.encoder,
                                           relation_size=relation_size,
                                           shaking_type=self.args.shaking_type,
                                           inner_enc_type="min_pooling",
                                           dist_emb_size=self.args.dist_emb_size,
                                           ent_add_dist=self.args.ent_add_dist,
                                           rel_add_dist=self.args.rel_add_dist
                                           )

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                                  help="loss type")
        model_parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                                  help="loss type")
        model_parser.add_argument("--loss_weight_recover_steps", type=int, default=10000)
        model_parser.add_argument("--shaking_type", type=str, default="cat",
                                  help="# cat, cat_plus, cln, cln_plus;"
                                       " Experiments show that cat/cat_plus work better with BiLSTM,"
                                       "while cln/cln_plus work better with BERT. The results in the paper are "
                                       "produced by cat. "
                                       "So, if you want to reproduce the results, cat is enough, no matter for BERT "
                                       "or BiLSTM.")
        model_parser.add_argument("--inner_enc_type", type=str, default=None,
                                  help="valid only if cat_plus or cln_plus is set."
                                       " It is the way how to encode inner tokens between each token pairs. "
                                       "If you only want to reproduce the results, just leave it alone.")
        model_parser.add_argument("--dist_emb_size", type=int, default=-1,
                                  help="do not use distance embedding; "
                                       "other number: need to be larger than the max_seq_len of the inputs."
                                       " set -1 if you only want to reproduce the results in the paper.")
        model_parser.add_argument("--ent_add_dist", action="store_true")
        model_parser.add_argument("--rel_add_dist", action="store_true")
        return model_parser

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.tplinker_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.tplinker_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.999),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon, )
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (
                self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(warmup_steps / t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total,
                                                                  lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        sample_list, batch_input_ids, \
        batch_attention_mask, batch_token_type_ids, \
        tok2char_span_list, batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch

        ent_shaking_outputs, \
        head_rel_shaking_outputs, \
        tail_rel_shaking_outputs = self.tplinker_model(batch_input_ids,
                                                       batch_attention_mask,
                                                       batch_token_type_ids,
                                                       )

        z = (2 * self.relation_size + 1)
        total_steps = self.args.loss_weight_recover_steps + 1  # + 1 avoid division by zero error
        current_step = self.trainer.global_step
        w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
        w_rel = min((self.relation_size / z) * current_step / total_steps,
                    (self.relation_size / z))

        loss = w_ent * self.compute_loss(ent_shaking_outputs, batch_ent_shaking_tag) + \
               w_rel * self.compute_loss(head_rel_shaking_outputs, batch_head_rel_shaking_tag) + \
               w_rel * self.compute_loss(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)
        self.log("loss", loss, prog_bar=True)
        return loss

    def compute_loss(self, pred, target, weights=None):
        if weights is not None:
            weights = torch.tensor(weights, dtype=float)
        cross_en = nn.CrossEntropyLoss(weight=weights)
        return cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))


def prepare_data():
    # 预处理训练数据，并且保存对应的文件（目前只针对训练数据）
    # 创建缓存文件的目录
    data_dir = "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/open-ie"
    cache_dir = "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/NlpTaskSpace/data/tplinker/cache"
    tokenizer = BertTokenizerFast.from_pretrained(
        "/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model")
    get_tok2char_span_map = lambda text: \
        tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    preprocessor = Preprocessor(tokenize_func=tokenizer.tokenize, get_tok2char_span_map_func=get_tok2char_span_map)
    print("prepare data first time")
    train_data = json.load(open(os.path.join(data_dir, "train_triples.json"), encoding="utf-8"))
    data = preprocessor.transform_data(train_data, ori_format="casrel", dataset_type="train",
                                       add_id=True)

    def check_tok_span(data):
        def extr_ent(text, tok_span, tok2char_span):
            char_span_list = tok2char_span[tok_span[0]:tok_span[1]]
            char_span = (char_span_list[0][0], char_span_list[-1][1])
            decoded_ent = text[char_span[0]:char_span[1]]
            return decoded_ent

        span_error_memory = set()
        for sample in tqdm(data, desc="check tok spans"):
            text = sample["text"]
            tok2char_span = get_tok2char_span_map(text)
            for ent in sample["entity_list"]:
                tok_span = ent["tok_span"]
                if extr_ent(text, tok_span, tok2char_span) != ent["text"]:
                    span_error_memory.add(
                        "extr ent: {}---gold ent: {}".format(extr_ent(text, tok_span, tok2char_span),
                                                             ent["text"]))

            for rel in sample["relation_list"]:
                subj_tok_span, obj_tok_span = rel["subj_tok_span"], rel["obj_tok_span"]
                if extr_ent(text, subj_tok_span, tok2char_span) != rel["subject"]:
                    span_error_memory.add(
                        "extr: {}---gold: {}".format(extr_ent(text, subj_tok_span, tok2char_span),
                                                     rel["subject"]))
                if extr_ent(text, obj_tok_span, tok2char_span) != rel["object"]:
                    span_error_memory.add(
                        "extr: {}---gold: {}".format(extr_ent(text, obj_tok_span, tok2char_span),
                                                     rel["object"]))

        return span_error_memory

    rel_set = set()
    ent_set = set()
    add_char_span = True
    if "relation_list" in data[0]:  # train or valid data
        # rm redundant whitespaces
        # separate by whitespaces
        data = preprocessor.clean_data_wo_span(data, separate=False)
        #         if file_name != "train_data":
        #             set_trace()
        # add char span
        if add_char_span:
            data, miss_sample_list = preprocessor.add_char_span(data, ignore_subword_match=True)
            print(";;;; mis sample list", len(miss_sample_list))
        #         # clean
        #         data, bad_samples_w_char_span_error = preprocessor.clean_data_w_span(data)
        #         error_statistics[file_name]["char_span_error"] = len(bad_samples_w_char_span_error)

        # collect relation types and entity types
        for sample in tqdm(data, desc="building relation type set and entity type set"):
            if "entity_list" not in sample:  # if "entity_list" not in sample, generate entity list with default type
                ent_list = []
                for rel in sample["relation_list"]:
                    ent_list.append({
                        "text": rel["subject"],
                        "type": "DEFAULT",
                        "char_span": rel["subj_char_span"],
                    })
                    ent_list.append({
                        "text": rel["object"],
                        "type": "DEFAULT",
                        "char_span": rel["obj_char_span"],
                    })
                sample["entity_list"] = ent_list

            for ent in sample["entity_list"]:
                ent_set.add(ent["type"])

            for rel in sample["relation_list"]:
                rel_set.add(rel["predicate"])

        # add tok span
        data = preprocessor.add_tok_span(data)

        # check tok span
        if True:
            span_error_memory = check_tok_span(data)
            if len(span_error_memory) > 0:
                print(span_error_memory)

    rel_set = sorted(rel_set)
    rel2id = {rel: ind for ind, rel in enumerate(rel_set)}

    ent_set = sorted(ent_set)
    ent2id = {ent: ind for ind, ent in enumerate(ent_set)}

    data_path = os.path.join(cache_dir, "train.json")
    json.dump(data, open(data_path, "w", encoding="utf-8"), ensure_ascii=False)

    rel2id_path = os.path.join(cache_dir, "rel2id.json")
    json.dump(rel2id, open(rel2id_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(",,,,数据集中的关系数目", len(rel2id))
    ent2id_path = os.path.join(cache_dir, "ent2id.json")
    json.dump(ent2id, open(ent2id_path, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")

    parser.add_argument("--data_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/NlpTaskSpace/data/tplinker",
                        help="data dir")
    parser.add_argument("--bert_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
                        help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
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
    parser = TplinkerExtraction.add_model_specific_args(parser)
    parser = TplinkerDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    check_point = ModelCheckpoint(dirpath=args.output_dir)
    # early_stop = EarlyStopping("f1", mode="max", patience=3, min_delta=0.2)
    trainer = Trainer.from_argparse_args(parser, default_root_dir=args.output_dir, callbacks=[check_point])
    data_module = TplinkerDataModule(args)
    model = TplinkerExtraction(args, relation_size=data_module.relation_size)
    trainer.fit(model, datamodule=data_module)
