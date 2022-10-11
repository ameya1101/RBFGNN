import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from layers import SAGPool
from torch_scatter import scatter_mean, scatter_std


class RBFGNN(torch.nn.Module):
    def __init__(self, args) -> None:
        super(RBFGNN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.num_hidden = args.num_hidden
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        conv = getattr(torch_geometric.nn, args.conv)

        self.conv1 = conv(self.num_features, self.num_hidden)
        self.pool1 = SAGPool(self.num_hidden, ratio=self.pooling_ratio)
        self.conv2 = conv(self.num_hidden, self.num_hidden)
        self.pool2 = SAGPool(self.num_hidden, ratio=self.pooling_ratio)
        self.conv3 = conv(self.num_hidden, self.num_hidden)
        self.pool3 = SAGPool(self.num_hidden, ratio=self.pooling_ratio)

        self.classifer = torch.nn.Sequential(
            [
                torch.nn.Linear(self.num_hidden * 2, self.num_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_ratio),
                torch.nn.Linear(self.num_hidden, self.num_hidden // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.num_hidden // 2, self.num_classes),
            ]
        )

    def mu(self, x, batch):
        return scatter_mean(x, batch.unsqueeze(1), dim=0)

    def var(self, x, batch):
        return torch.square(scatter_std(x, batch.unsqueeze(1), dim=0))

    def rbf_update(self, x, mu, var, batch):
      counts = torch.bincount(batch)
      mu = torch.repeat_interleave(mu, counts, dim=0)
      var = torch.repeat_interleave(var, counts, dim=0)
      return (1.0 / (torch.sqrt(2 * torch.pi * var) + 1e-6)) * torch.exp(
          -0.5 * (x - mu) ** 2 / var
        )


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x = self.rbf_update(x, self.mu(x, batch), self.var(x, batch), batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x = self.rbf_update(x, self.mu(x, batch), self.var(x, batch), batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x = self.rbf_update(x, self.mu(x, batch), self.var(x, batch), batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)


        x = x1 + x2 + x3

        x = self.classifer(x)
        return F.log_softmax(x, dim=-1)
