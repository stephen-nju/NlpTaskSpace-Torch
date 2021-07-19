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
from typing import Optional, Any, List, Union
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

    def get_indexed_data(self, data, max_seq_len, data_type="test"):
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

    def generate_batch(self, batch_data, data_type="test"):
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

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=TplinkerDataset(self.train_feature),
                          batch_size=self.batch_size,
                          num_workers=4,
                          collate_fn=self.generate_batch,
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        pred_sample_list = []
        sample_list, batch_input_ids, \
        batch_attention_mask, batch_token_type_ids, \
        tok2char_span_list, batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch

        batch_ent_shaking_outputs, \
        batch_head_rel_shaking_outputs, \
        batch_tail_rel_shaking_outputs = self.tplinker_model(batch_input_ids,
                                                             batch_attention_mask,
                                                             batch_token_type_ids,
                                                             )
        batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_head_rel_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_tail_rel_shaking_outputs, dim=-1)

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            ent_shaking_tag, \
            head_rel_shaking_tag, \
            tail_rel_shaking_tag = batch_ent_shaking_tag[ind], \
                                   batch_head_rel_shaking_tag[ind], \
                                   batch_tail_rel_shaking_tag[ind]

            tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                         ent_shaking_tag,
                                                                         head_rel_shaking_tag,
                                                                         tail_rel_shaking_tag,
                                                                         tok2char_span,
                                                                         tok_offset=tok_offset,
                                                                         char_offset=char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
            })

            print(pred_sample_list)
            return pred_sample_list


# def get_test_prf(pred_sample_list, gold_test_data, pattern="only_head_text"):
#     text_id2gold_n_pred = {}
#     for sample in gold_test_data:
#         text_id = sample["id"]
#         text_id2gold_n_pred[text_id] = {
#             "gold_relation_list": sample["relation_list"],
#         }
#
#     for sample in pred_sample_list:
#         text_id = sample["id"]
#         text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]
#
#     correct_num, pred_num, gold_num = 0, 0, 0
#     for gold_n_pred in text_id2gold_n_pred.values():
#         gold_rel_list = gold_n_pred["gold_relation_list"]
#         pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
#         if pattern == "only_head_index":
#             gold_rel_set = set(
#                 ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
#                  in gold_rel_list])
#             pred_rel_set = set(
#                 ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
#                  in pred_rel_list])
#         elif pattern == "whole_span":
#             gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
#                                                                             rel["subj_tok_span"][1], rel["predicate"],
#                                                                             rel["obj_tok_span"][0],
#                                                                             rel["obj_tok_span"][1]) for rel in
#                                 gold_rel_list])
#             pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
#                                                                             rel["subj_tok_span"][1], rel["predicate"],
#                                                                             rel["obj_tok_span"][0],
#                                                                             rel["obj_tok_span"][1]) for rel in
#                                 pred_rel_list])
#         elif pattern == "whole_text":
#             gold_rel_set = set(
#                 ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
#             pred_rel_set = set(
#                 ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
#         elif pattern == "only_head_text":
#             gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
#                                                             rel["object"].split(" ")[0]) for rel in gold_rel_list])
#             pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
#                                                             rel["object"].split(" ")[0]) for rel in pred_rel_list])
#
#         for rel_str in pred_rel_set:
#             if rel_str in gold_rel_set:
#                 correct_num += 1
#
#         pred_num += len(pred_rel_set)
#         gold_num += len(gold_rel_set)
#     #     print((correct_num, pred_num, gold_num))
#     # prf = metrics.get_prf_scores(correct_num, pred_num, gold_num)
#     return prf


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
    trainer = Trainer.from_argparse_args(parser, default_root_dir=args.output_dir, callbacks=[check_point])
    data_module = TplinkerDataModule(args)
    model = TplinkerExtraction(args, relation_size=data_module.relation_size)
    CHECKPOINTS = "./output_dir/epoch=0-step=338.ckpt"
    hparams_file = "./output_dir/lightning_logs/version_7/hparams.yaml"
    model = model.load_from_checkpoint(
        hparams_file=hparams_file,
        checkpoint_path=CHECKPOINTS)
    trainer.predict(model, datamodule=data_module)
