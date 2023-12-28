from typing import Callable, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy.io as scio

from torch_geometric.utils import coalesce, scatter, softmax


class EdgePooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_sigmoid
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)



    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        edge_score = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_score = F.relu(self.lin(edge_score))
        edge_score = F.dropout(edge_score, p=self.dropout, training=self.training)
        edge_score = self.compute_edge_score(edge_score, edge_index, x.size(0))
        new_edge_index, new_edge_score = self.edge_delete_adap(batch, edge_score)
      
        return x, new_edge_index, batch, new_edge_score


    def edge_delete_adap(
            self,
            batch: Tensor,
            edge_score: Tensor,
    ):
        edges = len(edge_score) / (max(batch) + 1)
        nodes = int(torch.sqrt(edges))
        edges = torch.tensor(edges.ceil(), dtype=torch.long)
        edge_score = torch.reshape(edge_score, [max(batch) + 1, edges])
        rate = torch.rand_like(edge_score)
        perm = torch.where(rate > (1 - edge_score), torch.tensor(1).cuda(), torch.tensor(0).cuda())
        pos = torch.nonzero(perm)
        new_edge_index0 = torch.div(pos[:, 1], nodes, rounding_mode='trunc') + pos[:, 0] * nodes
        new_edge_index1 = torch.remainder(pos[:, 1], nodes) + pos[:, 0] * nodes
        new_edge_index = torch.concat((new_edge_index0.unsqueeze(0), new_edge_index1.unsqueeze(0)), dim=0)

        return new_edge_index, edge_score