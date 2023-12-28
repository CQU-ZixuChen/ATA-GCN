import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from edge_pool import EdgePooling
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.device = 'cuda'
if torch.cuda.is_available():
    args.device = 'cuda:0'



class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # XJTU
        self.conv1 = ChebConv(1025, 512, 2)
        self.pool1 = EdgePooling(in_channels=1025)
        self.conv2 = ChebConv(512, 512, 2)
        self.pool2 = EdgePooling(in_channels=512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(512)

        self.lin1 = torch.nn.Linear(1024, 512)
        self.lin2 = torch.nn.Linear(512, 256)
        self.lin3 = torch.nn.Linear(256, 4)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.bn4 = torch.nn.BatchNorm1d(256)

    def forward(self, data):
        x, edge_index, batch, A = data.x, data.edge_index, data.batch, data.A

        x, edge_index, batch, edge_score = self.pool1(x=x, edge_index=edge_index, batch=batch)
        edge_index = edge_index.to(args.device)

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Class-wise output
        x = F.relu(self.bn3(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bn4(self.lin2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        feature = x
        x = F.relu(self.lin3(x))
        output = F.log_softmax(x, dim=-1)

        return output, feature

