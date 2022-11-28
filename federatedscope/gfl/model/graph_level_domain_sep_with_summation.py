from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.utils import degree
from torch_geometric.utils import negative_sampling
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, \
    global_max_pool
from wandb.util import np

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.gine import GINE_Net
from federatedscope.gfl.model.gine_no_jk import GINE_NO_JK_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net
# graph_level_domain_sep_gine
EPS = 1e-15
EMD_DIM = 200


class AtomEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i in range(in_channels):
            emb = torch.nn.Embedding(EMD_DIM, hidden)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class GNN_Net_Graph(torch.nn.Module):
    r"""GNN model with pre-linear layer, pooling layer
        and output layer for graph classification tasks.
    Arguments:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden (int): hidden dim for all modules.
        max_depth (int): number of layers for gnn.
        dropout (float): dropout probability.
        gnn (str): name of gnn type, use ("gcn" or "gin").
        pooling (str): pooling method, use ("add", "mean" or "max").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 gnn='gcn',
                 pooling='add',
                 edge_dim = None):
        super(GNN_Net_Graph, self).__init__()
        self.dropout = dropout
        # Embedding (pre) layer
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        # GNN layer
        if gnn == 'gcn':
            self.gnn = GCN_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
        elif gnn == 'sage':
            self.gnn = SAGE_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)
        elif gnn == 'gat':
            self.gnn = GAT_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
        elif gnn == 'gin':
            self.local_gnn = GINE_NO_JK_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
            self.global_gnn = GINE_NO_JK_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)
        elif gnn == 'gpr':
            self.gnn = GPR_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               K=max_depth,
                               dropout=dropout)
        else:
            raise ValueError(f'Unsupported gnn type: {gnn}.')

        # Pooling layer
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(f'Unsupported pooling type: {pooling}.')
        # Output layer
        self.linear_out1_glob = Linear(hidden*max_depth, hidden)
        self.linear_out2_glob = Sequential(Linear(hidden, 64))
        self.bn_linear1_glob = BatchNorm1d(hidden)
        self.bn_linear2_glob = BatchNorm1d(64)
        self.linear_out1_loc = Linear(hidden * max_depth, hidden)
        self.linear_out2_loc = Sequential(Linear(hidden, 64))
        self.bn_linear1_loc = BatchNorm1d(hidden)
        self.bn_linear2_loc = BatchNorm1d(64)
        self.clf = Linear(64, out_channels)
        self.emb = Linear(edge_dim, hidden)
        torch.nn.init.xavier_normal_(self.emb.weight.data)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.get('edge_attr')
        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), 1).float()

        x = self.encoder(x)

        edge_attr = self.emb(edge_attr)
        x_local = self.local_gnn(x, edge_index, edge_attr)
        x_global = self.global_gnn(x, edge_index, edge_attr)

        x_local = self.pooling(x_local, batch)
        x_global = self.pooling(x_global, batch)

        x_global = self.linear_out1_glob(x_global)
        x_global = self.bn_linear1_glob(x_global)
        x_global = self.linear_out2_glob(x_global)
        x_global = self.bn_linear2_glob(x_global)
        x_global = x_global.relu()

        x_local = self.linear_out1_loc(x_local)
        x_local = self.bn_linear1_loc(x_local)
        x_local = self.linear_out2_loc(x_local)
        x_local = self.bn_linear2_loc(x_local)
        x_local = x_local.relu()

        x = x_local + x_global


        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x, x_local, x_global

