# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
from zamr.utils.gelu import GELU


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, graph_type='rectified', gcn_gate=True):
        super(GCNLayer, self).__init__()
        self.rec_fc_in = nn.Linear(in_dim, out_dim)
        self.rec_fc_out = nn.Linear(in_dim, out_dim)

        self.dep_fc_in = nn.Linear(in_dim, out_dim)
        self.dep_fc_out = nn.Linear(in_dim, out_dim)

        self.gcn_gate = gcn_gate

        self.gelu = GELU()

        self.refine_gate = Parameter(torch.randn(in_dim, 1))
        nn.init.xavier_normal_(self.refine_gate)
        self.layer_norm = nn.LayerNorm(out_dim * 2)

        if self.gcn_gate:
            self.rec_gate_W_in = Parameter(torch.randn(in_dim, 1))
            nn.init.xavier_normal_(self.rec_gate_W_in)
            self.rec_gate_b_in = Parameter(torch.randn(1))
            nn.init.zeros_(self.rec_gate_b_in)
            self.rec_gate_W_out = Parameter(torch.randn(in_dim, 1))
            nn.init.xavier_normal_(self.rec_gate_W_out)
            self.rec_gate_b_out = Parameter(torch.randn(1))
            nn.init.zeros_(self.rec_gate_b_out)

    def forward(self, gcn_input, latent_adj, dep_adj):
        latent_adj_out = torch.transpose(latent_adj, -1, -2)
        dep_adj_out = torch.transpose(dep_adj, -1, -2)

        refine_gate = torch.sigmoid(torch.tensordot(gcn_input, self.refine_gate, dims=([2], [0])))

        rec_adj_in = refine_gate * dep_adj + (torch.ones_like(refine_gate) - refine_gate) * latent_adj
        rec_adj_out = refine_gate * dep_adj_out + (torch.ones_like(refine_gate) - refine_gate) * latent_adj_out

        rec_Ax_in = rec_adj_in.bmm(gcn_input)
        rec_Ax_out = rec_adj_out.bmm(gcn_input)

        if self.gcn_gate:
            # latent
            rec_X_gate_in = torch.sigmoid(
                torch.tensordot(rec_Ax_in, self.rec_gate_W_in, dims=([2], [0])))
            rec_Ax_in = rec_Ax_in * rec_X_gate_in + self.rec_gate_b_in
            rec_X_gate_out = torch.sigmoid(
                torch.tensordot(rec_Ax_out, self.rec_gate_W_out, dims=([2], [0])))
            rec_Ax_out = rec_Ax_out * rec_X_gate_out + self.rec_gate_b_out

        rec_AxW_in = self.rec_fc_in(rec_Ax_in)
        rec_AxW_out = self.rec_fc_out(rec_Ax_out)

        # Self loop latent
        rec_AxW_in = rec_AxW_in + self.rec_fc_in(gcn_input)
        rec_AxW_out = rec_AxW_out + self.rec_fc_out(gcn_input)

        AxW = torch.cat((rec_AxW_in, rec_AxW_out), -1)
        AxW = self.layer_norm(AxW)
        return self.gelu(AxW), rec_adj_in


class RecGCN(nn.Module):
    def __init__(self, params, graph_type, input_dim, c_gcn=False):
        super(RecGCN, self).__init__()
        self.num_layers = params['n_layers']
        self.dropout = params['dropout']
        self.hidden_dim = params['gcn_hidden']
        self.rnn_dim = params['rnn_hidden']
        self.rnn_layer = params['rnn_layer']

        self.in_dim = input_dim

        self.graph_type = graph_type
        self.c_gcn = c_gcn
        self.residual = params['residual']

        # Use gate or not
        self.gcn_gate = params['gcn_gate']

        # C-GCN
        if self.c_gcn:
            self.rnn = nn.LSTM(self.in_dim, self.rnn_dim, self.rnn_layer, batch_first=True,
                               dropout=0, bidirectional=True)
            self.in_dim = 2 * self.rnn_dim

        self.gcn_layer = nn.ModuleList()
        for layer in range(self.num_layers):
            input_dim = self.in_dim if layer == 0 else self.hidden_dim * 2
            self.gcn_layer.append(GCNLayer(input_dim, self.hidden_dim, self.graph_type, self.gcn_gate))

    def rnn_encoder(self, rnn_inputs, mask, batch_size):
        seq_lens = torch.tensor(list(mask.data.eq(1).long().sum(1).squeeze()))
        h0, c0 = rnn_zero_state(batch_size, self.rnn_dim, self.rnn_layer)
        h0, c0 = h0.to(device=rnn_inputs.device), c0.to(device=rnn_inputs.device)

        # Sort seq
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort, idx_unsort = idx_sort.to(device=rnn_inputs.device), idx_unsort.to(device=rnn_inputs.device)

        rnn_inputs = rnn_inputs.index_select(0, idx_sort)
        seq_lens = list(seq_lens[idx_sort])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        # Restore index
        rnn_outputs = rnn_outputs.index_select(0, idx_unsort)
        return rnn_outputs

    def forward(self, rec_adj, dep_adj, inputs, mask=None):

        # RNN layer
        if self.c_gcn:
            inputs = F.dropout(self.rnn_encoder(inputs, mask, inputs.size()[0]), self.dropout)
        else:
            inputs = inputs

        if self.residual:
            for i, gcn in enumerate(self.gcn_layer):
                if i == 0:
                    memory_bank = gcn(inputs, rec_adj, dep_adj)
                elif i == 1:
                    prev_mem_bank = inputs + memory_bank
                    memory_bank = gcn(prev_mem_bank, rec_adj, dep_adj)
                else:
                    prev_mem_bank = prev_mem_bank + memory_bank
                    memory_bank = gcn(prev_mem_bank, rec_adj, dep_adj)

                inputs = F.dropout(memory_bank, self.dropout) if i < self.num_layers - 1 else memory_bank
            return inputs



        else:
            for i, gcn in enumerate(self.gcn_layer):
                inputs = gcn(inputs, rec_adj, dep_adj)
                inputs = F.dropout(inputs, self.dropout) if i < self.num_layers - 1 else inputs

            return inputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = torch.zeros(*state_shape)
    return h0, c0
