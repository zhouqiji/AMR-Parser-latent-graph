# -*- coding: utf-8 -*-
from .explicit_graph_encoder import ExplicitGraphEncoder
from .latent_graph_encoder import LatentGraphEncoder
from .gcn import GCN
from .rec_gcn import RecGCN
from torch.nn import Parameter
from zamr.utils.gelu import GELU

import torch
import torch.nn as nn


class GraphEncoder(nn.Module):

    def __init__(self, params):

        super(GraphEncoder, self).__init__()
        # get parameters
        self.params = params
        self.dist_type = params['latent_encoder']['latent_type']

        self.inp_dim = params['input_size']

        self.graph_type = params['graph_type']

        self.gcn_hidden_size = params['gcn_hidden']

        self.c_gcn = params['c_gcn']
        self.dropout = params['dropout']

        self.latent_graph = None
        self.dependent_graph = None

        self.selection_loss = None
        self.gelu = GELU()

        # Get graph representations: latent, dependent or hybrid
        if self.graph_type == 'latent':
            self.latent_encoder = LatentGraphEncoder(params['latent_encoder'], self.inp_dim)
            self.explicit_encoder = ExplicitGraphEncoder()
        elif self.graph_type == 'dependent':
            self.explicit_encoder = ExplicitGraphEncoder()

        elif self.graph_type == 'rectified':
            self.latent_encoder = LatentGraphEncoder(params['latent_encoder'], self.inp_dim)
            self.explicit_encoder = ExplicitGraphEncoder()
        else:
            raise NotImplementedError("Please set (latent, ependent or hybrid) for graph type")

        # Build the graph encoder with different graph type
        if self.graph_type != 'rectified':
            self.gcn = GCN(params, self.graph_type, self.inp_dim, c_gcn=self.c_gcn)
        else:
            self.gcn = RecGCN(params, self.graph_type, self.inp_dim, c_gcn=self.c_gcn)

    def forward(self, input_seq, mask, dep_heads):

        if self.graph_type == 'rectified':
            latent_graph, self.selection_loss = self.latent_encoder(input_seq, mask)
            self.latent_graph = latent_graph
            self.dependent_graph = self.explicit_encoder(dep_heads)

        elif self.graph_type == 'latent':
            latent_graph, self.selection_loss = self.latent_encoder(input_seq, mask)
            self.latent_graph = latent_graph

            self.dependent_graph = self.explicit_encoder(dep_heads)

        else:
            self.dependent_graph = self.explicit_encoder(dep_heads)
            self.latent_graph = torch.zeros_like(self.dependent_graph)

        if self.graph_type == 'latent':

            output = self.gcn(self.latent_graph, input_seq, mask)

        elif self.graph_type == 'dependent':

            output = self.gcn(self.dependent_graph, input_seq, mask)

        else:  # rectified
            output, fused_graph = self.gcn(self.latent_graph, self.dependent_graph, input_seq, mask)
            self.latent_graph = fused_graph

        return output, self.latent_graph, self.dependent_graph

    @classmethod
    def from_params(cls, params):
        return cls(params)
