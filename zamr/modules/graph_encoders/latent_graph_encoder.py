# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn

from .kuma_attention import KumaAttention
from .kuma_self_attention_transformer import KumaSelfAttention
from .kuma_utils import get_z_counts


class LatentGraphEncoder(nn.Module):
    """Latent graph model with Hardkuma attention"""

    def __init__(self, params, input_dim):
        self.params = params
        super(LatentGraphEncoder, self).__init__()

        self.attn_type = params['latent_type']
        self.support = params['support']
        self.max_relative_distance = 11

        self.selection = params['selection_rate']

        self.rel_embedding = nn.Embedding(self.max_relative_distance * 2 + 1, 1)
        nn.init.xavier_normal_(self.rel_embedding.weight)
        self.attention = KumaSelfAttention(input_dim, self.params['hidden_size'], support=self.support,
                                           max_relative_distance=self.max_relative_distance, dist_type=self.attn_type,
                                           dist_embed=self.rel_embedding)
        # self.attention = KumaAttention(input_dim, self.params['hidden_size'], support=self.support)

        self.mask = None
        self.graph = None

        self.lagrange_lr = self.params['lagrange_lr']
        self.lagrange_alpha = self.params['lagrange_alpha']
        self.lambda_init = self.params['lambda_init']
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average

    def forward(self, input_seq, mask):
        # self.graph = self.attention(input_seq, input_seq)
        self.graph = self.attention(input_seq, input_seq, input_seq, mask)
        self.mask = mask
        self.graph = self._mask_padding(self.graph, mask.unsqueeze(1), 0.)
        self.graph = self._mask_padding(self.graph, mask.unsqueeze(2), 0.)
        loss, optional = self.get_selection_loss()
        return self.graph, loss

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        return torch.where(mask.byte(), x, x.new_full([1], value))

    def get_selection_loss(self):
        optional = OrderedDict()

        batch_size = self.graph.size(0)

        if self.training:
            z0, zc, z1 = get_z_counts(self.graph, mask=self.mask)
            zt = float(z0 + zc + z1)
            optional["p2h_0"] = z0 / zt
            optional["p2h_c"] = zc / zt
            optional["p2h_1"] = z1 / zt
            optional["p2h_selected"] = 1 - optional["p2h_0"]

        # only for hardkuma
        assert isinstance(self.attention, KumaSelfAttention) or isinstance(self.attention, KumaAttention), \
            "expected HK attention for this model, please set dist=hardkuma"

        # init the selection loss
        loss = 0

        if self.selection > 0:
            # kuma attention distribution (computed in forward call)
            z_dist = self.attention.dist

            pdf0 = z_dist.pdf(0.)
            pdf0 = pdf0.squeeze(-1)

            seq_lengths = self.mask.sum(1).float()

            # L0 regularizer

            # probability of being non-zero (masked for invalid positions)
            # we first mask all invalid positions in the tensor
            # first we mask invalid hypothesis positions
            #   (dim 2,broadcast over dim1)
            # then we mask invalid premise positions
            #   (dim 1, broadcast over dim 2)

            pdf_nonzero = 1. - pdf0  # [B, T]
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.mask.unsqueeze(1), value=0.)
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.mask.unsqueeze(2), value=0.)

            l0 = pdf_nonzero.sum(2) / (seq_lengths.unsqueeze(1) + 1e-9)
            l0 = l0.sum(1) / (seq_lengths + 1e-9)
            l0 = l0.sum() / batch_size

            # `l0` now has the expected selection rate for this mini-batch
            # we now follow the steps Algorithm 1 (page 7) of this paper:
            # https://arxiv.org/abs/1810.00597
            # to enforce the constraint that we want l0 to be not higher
            # than `self.selection` (the target sparsity rate)

            # lagrange dissatisfaction, batch average of the constraint
            c0_hat = (l0 - self.selection)

            # moving average of the constraint
            self.c0_ma = self.lagrange_alpha * self.c0_ma + (
                    1 - self.lagrange_alpha) * c0_hat.item()

            # compute smoothed constraint (equals moving average c0_ma)
            c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

            # update lambda
            self.lambda0 = self.lambda0 * torch.exp(
                self.lagrange_lr * c0.detach())

            with torch.no_grad():
                optional["cost0_l0"] = l0.item()
                optional["target0"] = self.selection
                optional["c0_hat"] = c0_hat.item()
                optional["c0"] = c0.item()  # same as moving average
                optional["lambda0"] = self.lambda0.item()
                optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
                optional["a"] = z_dist.a.mean().item()
                optional["b"] = z_dist.b.mean().item()
            loss = self.lambda0.detach() * c0

        return loss, optional
