import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch_geometric.nn import LayerNorm, TransformerConv


class TransformerGNN(torch.nn.Module):
    def __init__(self,
                 embedding_size=64,
                 edge_dim=None):
        super(TransformerGNN, self).__init__()
        edge_dim = None if edge_dim == 0 else edge_dim
        torch.manual_seed(42)
        # GCN layers
        print(f"edge_dim: {edge_dim}")
        self.conv1 = TransformerConv(embedding_size, embedding_size, edge_dim=edge_dim)
        self.ln2 = LayerNorm(embedding_size)
        self.topk_pool1 = TopKPooling(embedding_size, 0.5)
        self.conv2 = TransformerConv(embedding_size, embedding_size, edge_dim=edge_dim)
        self.ln3 = LayerNorm(embedding_size)
        self.topk_pool2 = TopKPooling(embedding_size, 0.5)
        self.conv3 = TransformerConv(embedding_size, embedding_size, edge_dim=edge_dim)
        self.ln4 = LayerNorm(embedding_size)
        self.topk_pool3 = TopKPooling(embedding_size, 0.5)

    def forward(self, x, edge_index, edge_attributes, batch_index):
        # First Conv layers
        if edge_attributes is not None:
            hidden = self.conv1(x, edge_index, edge_attributes)
        else:
            hidden = self.conv1(x, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln2(hidden)
        hidden, edge_index, edge_attr, batch, perm, score = self.topk_pool1(hidden, edge_index, None, batch_index)
        x1 = torch.cat([gmp(hidden, batch), gap(hidden, batch)], dim=1)
        if edge_attr is not None:
            hidden = self.conv2(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv2(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln3(hidden)
        hidden, edge_index, edge_attr, batch, perm, score = self.topk_pool2(hidden, edge_index, None, batch)
        # x2 = torch.cat([gmp(hidden, batch), gap(hidden, batch)], dim=1)
        # hidden = x1 + x2
        # Global Pooling layer
        # hidden = torch.cat([gmp(hidden, None),
        #                    gap(hidden, None)], dim=1)

        # Apply a final linear classifier
        # out = self.out1(hidden)

        return hidden, batch


