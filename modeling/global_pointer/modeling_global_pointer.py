from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch


def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)

def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits

class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False
    ):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        input_shape = inputs.shape
        _, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size, self.head_size * self.heads * 2)

#     def reset_params(self):
#         nn.init.xavier_uniform_(self.dense.weight)

    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2, dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size, hidden_size, RoPE=True):
        super(EfficientGlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense_1 = nn.Linear(hidden_size, self.head_size * 2)
        self.dense_2 = nn.Linear(self.head_size * 2, self.heads * 2)

    def forward(self, inputs, mask=None):
        inputs = self.dense_1(inputs)  # batch,
        # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # 分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmd , bnd -> bmn', qw, kw) / self.head_size ** 0.5
        bias = torch.einsum('bnh -> bhn', self.dense_2(inputs)) / 2
        logits = logits[:, None] + bias[:, :self.heads, None] + bias[:, self.heads:, :, None]
        # 排除padding 排除下三角
        logits = add_mask_tril(logits, mask)

        # scale返回
        return logits



class GlobalPointerNer(BertPreTrainedModel):
    """
    GlobalPointer + Bert 的命名实体模型

    Args:
        config: 模型的配置对象
        bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

    Reference:
        [1] https://www.kexue.fm/archives/8373
    """  # noqa: ignore flake8"
    


    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.global_pointer = GlobalPointer(
            self.num_labels,
            config.head_size,
            config.hidden_size
        )
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True
        ).hidden_states

        sequence_output = outputs[-1]

        logits = self.global_pointer(sequence_output, mask=attention_mask)

        return logits

# class EfficientGlobalPointerBert(BertPreTrainedModel):
#     """
#     EfficientGlobalPointer + Bert 的命名实体模型

#     Args:
#         config: 模型的配置对象
#         bert_trained (:obj:`bool`, optional): 预训练模型的参数是否可训练

#     Reference:
#         [1] https://www.kexue.fm/archives/8877
#         [2] https://github.com/powerycy/Efficient-GlobalPointer
#     """  # noqa: ignore flake8"

#     def __init__(
#         self,
#         config,
#         encoder_trained=True,
#         head_size=64
#     ):
#         super(EfficientGlobalPointerBert, self).__init__(config)

#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)

#         for param in self.bert.parameters():
#             param.requires_grad = encoder_trained

#         self.efficient_global_pointer = EfficientGlobalPointer(
#             self.num_labels,
#             head_size,
#             config.hidden_size
#         )

#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         **kwargs
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True,
#             output_hidden_states=True
#         ).hidden_states

#         sequence_output = outputs[-1]

#         logits = self.efficient_global_pointer(sequence_output, mask=attention_mask)

#         return logits
