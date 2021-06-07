import math

import torch as T
import torch.nn as nn


def glorot_uniform_init(weight, fan_in, fan_out):
    v = 6 if (fan_in != 0 and fan_out != 0) else 3
    bound = float(math.sqrt(v / (fan_in + fan_out)))
    nn.init.uniform_(weight, a=-bound, b=bound)


def generate_relative_positional_embeddings(max_len, d_model):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(2 * max_len + 1, d_model)
        position = T.arange(-max_len, max_len + 1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (2 * max_len + 1, d_model)
        pe = nn.Embedding.from_pretrained(pe,
                                          freeze=True)
    return pe


def generate_temporal_encodings(time, d_model):
    with T.no_grad():
        pe = T.zeros(d_model).float()
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[0::2] = T.sin(time * div_term)
        pe[1::2] = T.cos(time * div_term)

        pe = pe.view(1, 1, d_model)

    return pe


def generate_position_scores(qS, vS,
                             intermediate_position_scores,
                             scalar_position_embeddings,
                             cls2others_weight=None,
                             others2cls_weight=None):
    N, H, _, _ = intermediate_position_scores.size()
    assert intermediate_position_scores.size() == (N, H, qS, vS)

    # Relative Embedding
    S = max([qS, vS])
    positions = T.arange(S).to(scalar_position_embeddings.weight.device)
    positions = positions.unsqueeze(0).repeat(S, 1)
    positions_t = positions.permute(1, 0).contiguous()
    relative_position_idx = positions - positions_t

    assert relative_position_idx.size() == (S, S)

    M = scalar_position_embeddings.weight.size(0)
    K = (M - 1) // 2
    relative_position_idx = T.clamp(relative_position_idx, min=-K, max=K)
    relative_position_idx = relative_position_idx + K
    relative_position_idx = relative_position_idx[0:qS, 0:vS]
    relative_position_bias = scalar_position_embeddings(relative_position_idx)

    assert relative_position_bias.size() == (qS, vS, H)

    relative_position_bias = relative_position_bias.unsqueeze(0)
    relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()

    assert relative_position_bias.size() == (1, H, qS, vS)

    position_scores = relative_position_bias + intermediate_position_scores

    if cls2others_weight is not None and others2cls_weight is not None:
        position_scores = position_scores[..., 1:]
        others2cls_weight = others2cls_weight.view(1, H, 1, 1).repeat(N, 1, qS, 1)
        position_scores = T.cat([others2cls_weight, position_scores], dim=-1)
        position_scores = position_scores[..., 1:, :]
        cls2others_weight = cls2others_weight.view(1, H, 1, 1).repeat(N, 1, 1, vS)
        position_scores = T.cat([cls2others_weight, position_scores], dim=-2)

    return position_scores


def gelu(x):
    return 0.5 * x * (1 + T.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
