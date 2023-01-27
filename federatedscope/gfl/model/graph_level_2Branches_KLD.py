import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net

EMD_DIM = 200

# graph_level_2Branches_KLD
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
                 pooling='add'):
        super(GNN_Net_Graph, self).__init__()
        self.hidden = hidden
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
            self.local_gnn = GIN_Net(in_channels=hidden,
                                     out_channels=hidden,
                                     hidden=hidden,
                                     max_depth=max_depth,
                                     dropout=dropout)
            self.global_gnn = GIN_Net(in_channels=hidden,
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
        self.linear = Linear(hidden, hidden*2)
        self.clf = Linear(hidden, out_channels)

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def kld_gauss(self, u1, logvar1, u2, logvar2):
        # general KL two Gaussians
        # u2, s2 often N(0,1)
        # https://stats.stackexchange.com/questions/7440/ +
        # kl-divergence-between-two-univariate-gaussians
        # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
        s1 = logvar1.mul(0.5).exp_()
        s2 = logvar2.mul(0.5).exp_()
        v1 = s1 * s1
        v2 = s2 * s2
        a = torch.log(s2 / s1)
        num = v1 + (u1 - u2) ** 2
        den = 2 * v2
        b = num / den
        res = a + b - 0.5
        tmp = torch.mean(res)
        if tmp > 1:
            asdsad= 12321
        return torch.mean(res)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)


        # local encoder
        x_local = self.local_gnn((x, edge_index))
        x_local = self.pooling(x_local, batch)

        mu_logvar_local = self.linear(x_local).view(-1, 2, self.hidden)
        mu_local = mu_logvar_local[:, 0, :]
        logvar_local = mu_logvar_local[:, 1, :]
        x_local = self.reparameterise(mu_local, logvar_local)
        x_local = x_local.relu()

        # global encoder
        x_global = self.global_gnn((x, edge_index))
        x_global = self.pooling(x_global, batch)

        mu_logvar_global = self.linear(x_global).view(-1, 2, self.hidden)
        mu_global = mu_logvar_global[:, 0, :]
        logvar_global = mu_logvar_global[:, 1, :]
        x_global = self.reparameterise(mu_global, logvar_global)
        x_global = x_global.relu()

        x = x_local + x_global
        kld = self.kld_gauss(mu_local, logvar_local, mu_global, logvar_global)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x, kld
