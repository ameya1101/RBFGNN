from typing import List
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import torch


class SAGPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.8,
        conv=GCNConv,
        activation="Tanh",
    ) -> None:
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = conv(in_channels, 1)
        self.activation = getattr(torch.nn, activation)()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()
        if len(score.size()) == 0:
            score = score.unsqueeze(0)
        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.activation(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch, perm
