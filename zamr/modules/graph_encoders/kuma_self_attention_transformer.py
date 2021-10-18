# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn import Linear, Sequential, ReLU, Dropout

from .kuma_utils import get_relative_positions
from .kuma import Kuma, HardKuma

import numpy as np

MIN_CLAMP = 1e-3
MAX_CLAMP = 100


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = self._mask_padding(scores, attn_mask.unsqueeze(-1), 1e-9)
        scores = self._mask_padding(scores, attn_mask.unsqueeze(-2), 1e-9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        return torch.where(mask.byte(), x, x.new_full([1], value))


class KumaSelfAttention(nn.Module):
    def __init__(self, in_features, out_features, support=(-0.1, 1.1),
                 dropout=0.2, dist_type='hardkuma', add_rel_dist=True,
                 max_relative_distance=11, mask_diag=False, dist_embed=None):
        super(KumaSelfAttention, self).__init__()

        self.dist_type = dist_type
        self.activation = ReLU()
        self.dropout = Dropout(p=dropout)

        self.max_relative_distance = max_relative_distance
        self.mask_diag = mask_diag  # mask diagonal
        self.dist_embed = dist_embed
        self.add_rel_dist = add_rel_dist

        # For self attn
        self.d_model = in_features
        self.d_k = 64
        self.d_v = 64
        self.n_heads = 8

        self.a_W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.a_W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.a_W_V = nn.Linear(self.d_model, self.d_k * self.n_heads)

        self.b_W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.b_W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.b_W_V = nn.Linear(self.d_model, self.d_k * self.n_heads)

        self.a_attn = ScaledDotProductAttention(self.n_heads)
        self.b_attn = ScaledDotProductAttention(self.n_heads)

        self.a_score = nn.Linear(self.n_heads * self.d_v, self.n_heads * self.d_v)
        self.b_score = nn.Linear(self.n_heads * self.d_v, self.n_heads * self.d_v)

        self.layer_norm_a = nn.LayerNorm(self.n_heads * self.d_k)
        self.layer_norm_b = nn.LayerNorm(self.n_heads * self.d_k)

        self.support = support

        self.dist = None

    def _mask_diagnoal(self, x, mask_value=0.):
        """block the diagonal so a word does not self-align"""
        eye = torch.eye(x.size(1), dtype=torch.uint8, device=x.device)
        return torch.where(eye, x.new_full([1], mask_value), x)

    def _add_rel_dists(self, x):
        """add matrix of relative distances"""
        rel_dists = get_relative_positions(
            x.size(1), self.max_relative_distance, device=x.device)
        rel_dists = self.dist_embed(rel_dists).squeeze(-1).unsqueeze(0)
        return x + rel_dists

    def forward(self, Q, K, V, mask):

        batch_size = Q.size(0)

        a_q_s = self.a_W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        a_k_s = self.a_W_Q(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        a_v_s = self.a_W_Q(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        b_q_s = self.b_W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        b_k_s = self.b_W_Q(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        b_v_s = self.b_W_Q(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1)

        c_a, a_attn = self.a_attn(a_q_s, a_k_s, a_v_s, attn_mask)
        c_b, b_attn = self.b_attn(b_q_s, b_k_s, b_v_s, attn_mask)

        c_a = c_a.transpose(1, 2).contiguous().view(batch_size, -1,
                                                    self.n_heads * self.d_v)
        c_b = c_b.transpose(1, 2).contiguous().view(batch_size, -1,
                                                    self.n_heads * self.d_v)

        c_a = self.a_score(c_a) + c_a
        c_b = self.b_score(c_b) + c_b
        c_a = self.layer_norm_a(c_a)
        c_b = self.layer_norm_b(c_b)

        a = c_a @ c_a.transpose(-1, -2)
        b = c_b @ c_b.transpose(-1, -2)

        # norm
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()

        # add relative distances
        if self.add_rel_dist:
            a = self._add_rel_dists(a)
            b = self._add_rel_dists(b)

        a = softplus(a)
        b = softplus(b)

        a = a.clamp(MIN_CLAMP, MAX_CLAMP)

        b = b.clamp(MIN_CLAMP, MAX_CLAMP)

        # we return a distribution (from which we can sample if we want)
        if self.dist_type == "kuma":
            dist = Kuma([a, b])
        elif self.dist_type == "hardkuma":
            dist = HardKuma([a, b], support=self.support)
        else:
            raise ValueError("unknown dist")

        self.dist = dist

        if self.training:  # sample
            att = dist.sample()
        else:  # predict deterministically
            p0 = dist.pdf(Q.new_zeros(()))
            p1 = dist.pdf(Q.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            zero_one = torch.where(
                p0 > p1, Q.new_zeros([1]), Q.new_ones([1]))
            att = torch.where(pc < 0.5, zero_one, dist.mean())  # [B, M]

        if self.mask_diag:
            att = self._mask_diagnoal(att, mask_value=0.)

        return att  # [B, M]
