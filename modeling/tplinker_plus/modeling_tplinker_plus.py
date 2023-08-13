import math
import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel


class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解本文。
    参考：https://kexue.fm/archives/7359
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        """y_true ([Tensor]): [..., num_classes]
        y_pred ([Tensor]): [..., num_classes]
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = torch.cat(
            [y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1
        )
        y_pred_neg = torch.cat(
            [y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1
        )
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        return (pos_loss + neg_loss).mean()


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-12,
        conditional_size=False,
        weight=True,
        bias=True,
        norm_mode="normal",
        **kwargs
    ):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
        条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        # 兼容roformer_v2不包含weight
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        # 兼容t5不包含bias项, 和t5使用的RMSnorm
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm_mode = norm_mode
        self.eps = eps
        self.conditional_size = conditional_size
        if conditional_size:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, x):
        inputs = x[0]  # 这里是visible_hiddens
        if self.norm_mode == "rmsnorm":
            # t5使用的是RMSnorm
            variance = inputs.to(torch.float32).pow(2).mean(-1, keepdim=True)
            o = inputs * torch.rsqrt(variance + self.eps)
        else:
            # 归一化是针对于inputs
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            o = (inputs - u) / torch.sqrt(s + self.eps)
        if not hasattr(self, "weight"):
            self.weight = 1
        if not hasattr(self, "bias"):
            self.bias = 0
        if self.conditional_size:
            cond = x[1]  # 这里是repeat_hiddens
            # 三者的形状都是一致的
            # print(inputs.shape, cond.shape, o.shape)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            return (self.weight + self.dense1(cond)) * o + (
                self.bias + self.dense2(cond)
            )
        else:
            return self.weight * o + self.bias


class TplinkerHandshakingKernel(nn.Module):
    """
    Tplinker的HandshakingKernel实现
    """

    def __init__(self, hidden_size, shaking_type, inner_enc_type="lstm"):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(
                hidden_size, conditional_size=hidden_size
            )
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )
        # 自行实现的用torch.gather方式来做，避免循环，目前只实现了cat方式
        # tag_ids = [(i, j) for i in range(maxlen) for j in range(maxlen) if j >= i]
        # gather_idx = torch.tensor(tag_ids, dtype=torch.long).flatten()[None, :, None]
        # self.register_buffer('gather_idx', gather_idx)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = (
                    self.lamtha * torch.mean(seqence, dim=-2)
                    + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
                )
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [
                    pool(seq_hiddens[:, : i + 1, :], inner_enc_type)
                    for i in range(seq_hiddens.size()[1])
                ],
                dim=1,
            )
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        return inner_context

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]  # [batch_size, hidden_size]
            visible_hiddens = seq_hiddens[
                :, ind:, :
            ]  # ind: only look back, [batch_size, seq_len - ind, hidden_size]
            repeat_hiddens = hidden_each_step[:, None, :].repeat(
                1, seq_len - ind, 1
            )  # [batch_size, seq_len - ind, hidden_size]
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(
                    visible_hiddens, self.inner_enc_type
                )
                shaking_hiddens = torch.cat(
                    [repeat_hiddens, visible_hiddens, inner_context], dim=-1
                )
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(
                    visible_hiddens, self.inner_enc_type
                )
                shaking_hiddens = self.tp_cln([visible_hiddens, repeat_hiddens])
                shaking_hiddens = self.inner_context_cln(
                    [shaking_hiddens, inner_context]
                )
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


def upper_reg2seq(ori_tensor):
    """
    drop lower region and flat upper region to sequence
    :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
    :return: (batch_size, matrix_size + ... + 1, hidden_size)
    """

    tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
    uppder_ones = (
        torch.ones([tensor.size()[-1], tensor.size()[-1]])
        .long()
        .triu()
        .to(ori_tensor.device)
    )
    upper_diag_ids = torch.nonzero(uppder_ones.view(-1), as_tuple=False).view(-1)
    # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
    flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
    tensor_upper = torch.index_select(flat_tensor, dim=1, index=upper_diag_ids)
    return tensor_upper


class TplinkerHandshakingKernelPlus(nn.Module):
    def __init__(self, hidden_size, shaking_type, only_look_after=True):
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after
        if "cat" in shaking_type:
            self.cat_fc = nn.Linear(hidden_size * 2, hidden_size)
        if "cln" in shaking_type:
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        if "lstm" in shaking_type:
            assert only_look_after is True
            self.lstm4span = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                bidirectional=False,
                batch_first=True,
            )

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size_x)
        return:
            if only look after:
                shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size); e.g. (32, 5+4+3+2+1, 5)
            else:
                shaking_hiddenss: (batch_size, seq_len * seq_len, hidden_size)
        """
        seq_len = seq_hiddens.size()[1]
        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)
        shaking_pre = None

        # pre_num = 0
        def add_presentation(all_prst, prst):
            if all_prst is None:
                all_prst = prst
            else:
                all_prst += prst
            return all_prst

        if self.only_look_after:
            if "lstm" in self.shaking_type:
                batch_size, _, matrix_size, vis_hidden_size = visible.size()
                # mask lower triangle
                upper_visible = (
                    visible.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()
                )
                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = upper_visible.view(-1, matrix_size, vis_hidden_size)
                span_pre, _ = self.lstm4span(visible4lstm)
                span_pre = span_pre.view(
                    batch_size, matrix_size, matrix_size, vis_hidden_size
                )
                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = upper_reg2seq(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)
            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = upper_reg2seq(guide)
            visible = upper_reg2seq(visible)
        if "cat" in self.shaking_type:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.cat_fc(tp_cat_pre))
            shaking_pre = add_presentation(shaking_pre, tp_cat_pre)
        if "cln" in self.shaking_type:
            tp_cln_pre = self.tp_cln(visible, guide)
            shaking_pre = add_presentation(shaking_pre, tp_cln_pre)
        return shaking_pre


class TplinkerPlusNer(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super(TplinkerPlusNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.tok_pair_sample_rate = config.tok_pair_sample_rate
        self.dropout = nn.Dropout(classifier_dropout)
        self.handshaking_kernel = TplinkerHandshakingKernel(
            768, shaking_type="cat", inner_enc_type=""
        )
        self.fc = nn.Linear(768, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        is_training=True,
    ):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True,
        ).last_hidden_state
        shaking_hiddens = self.handshaking_kernel(bert_outputs)
        # print(f"output_shape==={bert_outputs.shape}")
        sampled_tok_pair_indices = None
        if is_training:
            # randomly sample segments of token pairs
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            # sampled_tok_pair_indices: (batch_size, ~segment_len) ~end_ind - start_ind <= segment_len
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(
                shaking_hiddens.size()[0], 1
            )
            #             sampled_tok_pair_indices = torch.randint(shaking_seq_len, (shaking_hiddens.size()[0], segment_len))
            sampled_tok_pair_indices = sampled_tok_pair_indices.to(
                shaking_hiddens.device
            )
            # sampled_tok_pair_indices will tell model what token pairs should be fed into fcs
            # shaking_hiddens: (batch_size, ~segment_len, hidden_size)
            shaking_hiddens = shaking_hiddens.gather(
                1,
                sampled_tok_pair_indices[:, :, None].repeat(
                    1, 1, shaking_hiddens.size()[-1]
                ),
            )

        # outputs: (batch_size, segment_len, tag_size) or (batch_size, shaking_seq_len, tag_size)
        output = self.fc(shaking_hiddens)  # [btz, pair_len, tag_size]
        if is_training:
            # print(f"output=={output.requires_grad}")
            return output, sampled_tok_pair_indices
        return output
