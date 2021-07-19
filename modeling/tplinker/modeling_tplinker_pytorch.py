# -*- coding: utf-8 -*-
"""
@author: zhubin
@email:  18073701@suning.com
@software: PyCharm
@file: modeling_tplinker.py
@time: 2021/5/13 14:56
"""
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch.nn.init import xavier_uniform_


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢?
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional=True)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size,
                                              hidden_size,
                                              num_layers=1,
                                              bidirectional=False,
                                              batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class TPLinkerBert(nn.Module):
    def __init__(self, encoder,
                 relation_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist
                 ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(relation_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(relation_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        #         if self.dist_emb_size != -1 and self.ent_add_dist:
        #             shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
        #         else:
        #             shaking_hiddens4ent = shaking_hiddens
        #         if self.dist_emb_size != -1 and self.rel_add_dist:
        #             shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None,:,:].repeat(shaking_hiddens.size()[0], 1, 1)
        #         else:
        #             shaking_hiddens4rel = shaking_hiddens

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class TPLinkerBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix,
                 emb_dropout_rate,
                 enc_hidden_size,
                 dec_hidden_size,
                 rnn_dropout_rate,
                 rel_size,
                 shaking_type,
                 inner_enc_type,
                 dist_emb_size,
                 ent_add_dist,
                 rel_add_dist):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1],
                                enc_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_size,
                                dec_hidden_size // 2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)

        hidden_size = dec_hidden_size

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]

        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, shaking_type, inner_enc_type)

        # distance embedding
        self.dist_emb_size = dist_emb_size
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = ent_add_dist
        self.rel_add_dist = rel_add_dist

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # add distance embeddings if it is set
        if self.dist_emb_size != -1:
            # set self.dist_embbedings
            hidden_size = shaking_hiddens.size()[-1]
            if self.dist_embbedings is None:
                dist_emb = torch.zeros([self.dist_emb_size, hidden_size]).to(shaking_hiddens.device)
                for d in range(self.dist_emb_size):
                    for i in range(hidden_size):
                        if i % 2 == 0:
                            dist_emb[d][i] = math.sin(d / 10000 ** (i / hidden_size))
                        else:
                            dist_emb[d][i] = math.cos(d / 10000 ** ((i - 1) / hidden_size))
                seq_len = input_ids.size()[1]
                dist_embbeding_segs = []
                for after_num in range(seq_len, 0, -1):
                    dist_embbeding_segs.append(dist_emb[:after_num, :])
                self.dist_embbedings = torch.cat(dist_embbeding_segs, dim=0)

            if self.ent_add_dist:
                shaking_hiddens4ent = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)
            if self.rel_add_dist:
                shaking_hiddens4rel = shaking_hiddens + self.dist_embbedings[None, :, :].repeat(
                    shaking_hiddens.size()[0], 1, 1)

        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)

        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))

        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class MetricsCalculator():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger

    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim=-1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim=1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc

    def get_rel_cpg(self, sample_list, tok2char_span_list,
                    batch_pred_ent_shaking_outputs,
                    batch_pred_head_rel_shaking_outputs,
                    batch_pred_tail_rel_shaking_outputs,
                    pattern="only_head_text"):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim=-1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim=-1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim=-1)

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                              pred_ent_shaking_tag,
                                                                              pred_head_rel_shaking_tag,
                                                                              pred_tail_rel_shaking_tag,
                                                                              tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                     rel in gold_rel_list])
                pred_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for
                     rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                                rel["subj_tok_span"][1],
                                                                                rel["predicate"],
                                                                                rel["obj_tok_span"][0],
                                                                                rel["obj_tok_span"][1]) for rel in
                                    pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                     gold_rel_list])
                pred_rel_set = set(
                    ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in
                     pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                                rel["object"].split(" ")[0]) for rel in pred_rel_list])

            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)

        return correct_num, pred_num, gold_num

    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1
