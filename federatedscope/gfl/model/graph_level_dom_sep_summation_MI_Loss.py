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

from federatedscope.gfl.model.MI_Network import Mine, T, MutualInformationEstimator
from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.gine import GINE_Net
from federatedscope.gfl.model.gine_no_jk import GINE_NO_JK_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net
# graph_level_dom_sep_summation_MI_Loss
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
                 edge_dim = None,
                 rho = 0.0):
        self.rho=rho
        print(f"rho: {rho}")
        if edge_dim is None or edge_dim == 0:
            edge_dim = 1
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
        #mi_model = T(hidden, hidden)

        #self.mine = Mine(mi_model, loss='mine')

        self.mine = MutualInformationEstimator(hidden, hidden, loss='mine')

        # Pooling layer
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(f'Unsupported pooling type: {pooling}.')

        self.bn_edge = BatchNorm1d(hidden)
        self.bn_node = BatchNorm1d(hidden)

        # Output layer
        self.global_linear_out1 = Linear(hidden * max_depth, hidden)
        #self.linear_out2_glob = Sequential(Linear(hidden, 64))
        self.bn_linear0_glob = BatchNorm1d(hidden * max_depth)
        self.bn_linear1_glob = BatchNorm1d(hidden)
        #self.bn_linear2_glob = BatchNorm1d(64)
        self.local_linear_out1 = Linear(hidden * max_depth, hidden)
        #self.linear_out2_loc = Sequential(Linear(hidden, 64))
        self.bn_linear0_loc = BatchNorm1d(hidden * max_depth)
        self.bn_linear1_loc = BatchNorm1d(hidden)
        self.bn_after_summation = BatchNorm1d(hidden)
        #self.bn_linear2_loc = BatchNorm1d(64)


        # local
        self.linear_out2 = Sequential(Linear(hidden, 64))
        self.bn_linear2 = BatchNorm1d(64)
        self.clf = Linear(64, out_channels)
        self.emb = Linear(edge_dim, hidden)
        #torch.nn.init.xavier_normal_(self.emb.weight.data)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.get('edge_attr')
        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), 1).float()
        else:
            edge_attr = edge_attr + 1

        x = self.encoder(x)
        x = self.bn_node(x)
        edge_attr = self.emb(edge_attr)
        edge_attr = self.bn_edge(edge_attr)

        x_local = self.local_gnn(x, edge_index, edge_attr)

        x_global = self.global_gnn(x, edge_index, edge_attr)

        x_global = self.pooling(x_global, batch)
        x_global = self.bn_linear0_glob(x_global)
        x_global = F.dropout(x_global, self.dropout, training=self.training)
        x_global = self.global_linear_out1(x_global).relu()


        #if(edge_attr.size(0)==3952):
        #    print(f"global: {x_global}")
        #x_global = self.linear_out2_glob(x_global).relu()
        #x_global = self.bn_linear2_glob(x_global)
        #x_global = F.dropout(x_global, self.dropout, training=self.training)

        x_local = self.pooling(x_local, batch)
        x_local = self.bn_linear0_loc(x_local)
        x_local = F.dropout(x_local, self.dropout, training=self.training)
        x_local = self.local_linear_out1(x_local).relu()

        #a = edge_attr.size(0)
        #b = edge_attr.size()
        #if (edge_attr.size(0) == 3952):
        #    print(f"local: {x_local}")
        #x_local = self.linear_out2_loc(x_local).relu()
        #x_local = self.bn_linear2_loc(x_local)
        #x_local = F.dropout(x_local, self.dropout, training=self.training)


        mi = self.mine(x_local, x_global)


        #x_global = self.bn_linear1_glob(x_global)
        #x_local = self.bn_linear1_loc(x_local)

        x = x_local + x_global

        x = self.bn_after_summation(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear_out2(x).relu()
        x = self.bn_linear2(x)
        x = self.clf(x)

        #return x, mi
        return x, mi

