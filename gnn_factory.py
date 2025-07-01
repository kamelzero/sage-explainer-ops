import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class SAGENodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, aggr: str = "mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr=aggr)
        self.conv2 = SAGEConv(hidden, hidden, aggr=aggr)
        self.lin   = nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)


def build_gnn(name: str, in_dim: int, hidden: int = 64, n_classes: int = 2):
    """
    Parameters
    ----------
    name : {"gcn", "sage"}
    in_dim : feature dimension
    hidden : hidden dim (same for all layers here)
    n_classes : output classes
    """
    name = name.lower()
    if name == "gcn":
        return GCNNodeClassifier(in_dim, hidden, n_classes)
    if name == "sage":
        return SAGENodeClassifier(in_dim, hidden, n_classes)
    raise ValueError(f"Unknown model '{name}'.  Choices: gcn, sage")
