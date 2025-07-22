from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from tqdm import tqdm

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_units, output_dim, K):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, hidden_units, K)
        self.linear = torch.nn.Linear(hidden_units, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.selu(h)
        h = self.linear(h)
        return h