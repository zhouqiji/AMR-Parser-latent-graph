# -*- coding: utf-8 -*-
import torch


class ExplicitGraphEncoder:
    def __init__(self):
        super(ExplicitGraphEncoder, self).__init__()
        pass

    def __call__(self, dep_heads):
        # Return the adj-matrix with the head information
        batch_size = dep_heads.size(0)
        max_len = dep_heads.size(1)
        graph = torch.zeros([batch_size, max_len, max_len])
        for i in range(batch_size):
            for j in range(max_len):
                dep = dep_heads[i][j]
                if dep != 0 and dep != -1:
                    graph[i][j][dep] = 1
        device = dep_heads.device
        return graph.to(device)
