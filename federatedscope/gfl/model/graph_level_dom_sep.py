from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from federatedscope.core.mlp import MLP
from torch.nn import Linear, Sequential, ModuleList, ParameterList
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import TransformerConv, LayerNorm, BatchNorm, GINConv
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, \
    global_max_pool
from torch_geometric.utils import degree
from torch_geometric.utils import negative_sampling
from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.gin_resnet import GIN_Res_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net

EPS = 1e-15
EMD_DIM = 200

"""
class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, edge_dim, dropout):
        super(NodeEncoder, self).__init__()
        self.dropout = dropout
        edge_dim = None if edge_dim == 0 else edge_dim
        self.conv1 = TransformerConv(in_channels, hidden, edge_dim=edge_dim)
        self.ln1 = LayerNorm(hidden)
        self.conv2 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln2 = LayerNorm(hidden)
        self.conv3 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln3 = LayerNorm(hidden)
        self.conv4 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln4 = LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            hidden = self.conv1(x, edge_index, edge_attr)
        else:
            hidden = self.conv1(x, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln1(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if edge_attr is not None:
            hidden = self.conv2(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv2(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln2(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden

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
        max_depth = 3
        print(f"dropout: {dropout}")
        self.dropout = dropout
        # Embedding (pre) layer
        self.bns = ModuleList()
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.node_encoder = NodeEncoder(in_channels, hidden, edge_dim, dropout)
        #self.encoder_atom = AtomEncoder(in_channels, hidden)
        #self.encoder = Linear(in_channels, hidden)
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
            self.gnn = GIN_Res_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
            #self.private_net = GIN_Res_Net(in_channels=hidden,
            #                   out_channels=hidden,
            #                   hidden=hidden,
            #                   max_depth=max_depth,
            #                   dropout=dropout)


        elif gnn == 'gin_res':
            self.gnn = GIN_Res_Net(in_channels=hidden,
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

        self.linear1 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        #self.layer_norm = torch.nn.LayerNorm(hidden)
        #seclflf.layer_norm2 = torch.nn.LayerNorm(hidden)

        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')
        x = self.node_encoder(x, edge_index, edge_attr)
        #if x.dtype == torch.int64:
        #    x = self.encoder_atom(x)
        #else:
        #    x = self.encoder(x)
        x = self.bns[0](x)
        #x = self.encoder2(x)
        #x = self.bns[1](x)
        #x = self.layer_norm(x)
        x = self.gnn((x, edge_index))
        #pr_x = self.private_net((x, edge_index))
        #x = x + enc
        #pr_x = self.pooling(x, batch)
        #pr_x = self.linear2(pr_x)
        x = self.pooling(x, batch)
        x = self.linear1(x)
        #x = torch.cat((x, pr_x), 1)

        #x = self.linear3(x)
        x = self.bns[1](x)
        #x = self.layer_norm2(x)
        x = F.dropout(x, self.dropout, training=self.training)

        #x = self.shared_linear2(x)
        #x = self.bns[3](x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.linear3(x)
        #x = self.linear3(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x


"""

"""
class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, edge_dim, dropout):
        super(NodeEncoder, self).__init__()
        self.dropout = dropout
        edge_dim = None if edge_dim == 0 else edge_dim
        self.conv1 = TransformerConv(in_channels, hidden, edge_dim=edge_dim)
        self.ln1 = LayerNorm(hidden)
        self.conv2 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln2 = LayerNorm(hidden)
        self.conv3 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln3 = LayerNorm(hidden)
        self.conv4 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln4 = LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            hidden = self.conv1(x, edge_index, edge_attr)
        else:
            hidden = self.conv1(x, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln1(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if edge_attr is not None:
            hidden = self.conv2(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv2(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln2(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if edge_attr is not None:
            hidden = self.conv3(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv3(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln3(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if edge_attr is not None:
            hidden = self.conv4(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv4(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln4(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden

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

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=5,
                 dropout=.0,
                 gnn='gcn',
                 pooling='add',
                 edge_dim = None):
        super(GNN_Net_Graph, self).__init__()
        max_depth = 3
        self.max_depth = 3
        print(f"dropout: {dropout}")
        print(f"max depth: {max_depth}")
        self.dropout = dropout
        # Embedding (pre) layer
        self.private_gnns = ModuleList()
        self.shared_gnns = ModuleList()
        self.private_bns = ModuleList()
        self.shared_bns = ModuleList()
        self.shared_net_alphas_s = ParameterList()
        self.shared_net_alphas_p = ParameterList()
        self.private_net_alphas_s = ParameterList()
        self.private_net_alphas_p = ParameterList()

        for i in range(max_depth):
            self.private_gnns.append(GINConv(MLP([hidden, hidden, hidden],
                                batch_norm=True)))
            self.shared_gnns.append(GINConv(MLP([hidden, hidden, hidden],
                                                 batch_norm=True)))
            self.private_bns.append(BatchNorm(hidden))
            self.shared_bns.append(BatchNorm(hidden))
            self.shared_net_alphas_s.append(torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True)))
            self.shared_net_alphas_p.append(torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True)))
            self.private_net_alphas_s.append(torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True)))
            self.private_net_alphas_p.append(torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True)))

        #self.shared_net_alpha = torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True))
        #self.private_net_alpha = torch.nn.Parameter(torch.full([64], 0.5, requires_grad=True))
        self.bns = ModuleList()
        self.node_encoder = NodeEncoder(in_channels, hidden, edge_dim, dropout)
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        self.encoder2 = Linear(hidden, hidden)
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

            self.gnn = GIN_Res_Net(in_channels=hidden,
                                   out_channels=hidden,
                                   hidden=hidden,
                                   max_depth=max_depth,
                                   dropout=dropout)
            self.private_net = GIN_Res_Net(in_channels=hidden,
                                   out_channels=hidden,
                                   hidden=hidden,
                                   max_depth=max_depth,
                                   dropout=dropout)

        elif gnn == 'gin_res':
            self.gnn = GIN_Res_Net(in_channels=hidden,
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
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        self.linear1 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        #self.layer_norm = torch.nn.LayerNorm(hidden)
        #self.layer_norm2 = torch.nn.LayerNorm(hidden)
        self.linear2 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.linear3 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')
        #x = self.node_encoder(x, edge_index, edge_attr)
        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)
        x = self.bns[0](x)
        #x = self.encoder2(x)
        #x = self.bns[1](x)
        #x = self.layer_norm(x)
        pr_x = x
        for i in range(self.max_depth):
            sh_x = self.shared_gnns[i](x, edge_index)
            sh_x=self.shared_bns[i](sh_x)
            sh_x = F.relu(F.dropout(sh_x, p=self.dropout, training=self.training))
            pr_x = self.shared_gnns[i](x, edge_index)
            pr_x = self.private_bns[i](pr_x)
            pr_x = F.relu(F.dropout(pr_x, p=self.dropout, training=self.training))

            sh_x = torch.mul(sh_x, self.shared_net_alphas_s[i]) + torch.mul(pr_x, self.shared_net_alphas_p[i])
            pr_x = torch.mul(sh_x, self.private_net_alphas_s[i]) + torch.mul(pr_x, self.private_net_alphas_p[i])


        #shared_x = self.gnn((x, edge_index))
        #pr_x = self.private_net((x, edge_index))
        #x = x + enc
        #pr_x = self.pooling(pr_x, batch)
        #pr_x = self.linear2(pr_x)
        x = torch.mul(sh_x, self.private_net_alphas_s[-1]) + torch.mul(pr_x, self.private_net_alphas_p[-1])
        x = self.pooling(pr_x, batch)# + self.pooling(x, batch)
        #x = self.linear1(x)
        #x = torch.mul(x, self.shared_net_alpha) + torch.mul(pr_x, self.private_net_alpha)
        #print(self.shared_net_alpha)
        #print(self.private_net_alpha)
        #x = torch.cat((x, pr_x), 1)

        x = self.linear3(x)
        x = self.bns[1](x)
        #x = self.layer_norm2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(F.dropout(x, self.dropout, training=self.training))

        #x = self.shared_linear2(x)
        #x = self.bns[3](x)
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = self.linear3(x)
        #x = self.linear3(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x
"""


class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, edge_dim, dropout):
        super(NodeEncoder, self).__init__()
        self.dropout = dropout
        edge_dim = None if edge_dim == 0 else edge_dim
        self.conv1 = TransformerConv(in_channels, hidden, edge_dim=edge_dim)
        self.ln1 = LayerNorm(hidden)
        self.conv2 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln2 = LayerNorm(hidden)
        self.conv3 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln3 = LayerNorm(hidden)
        self.conv4 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln4 = LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is not None:
            hidden = self.conv1(x, edge_index, edge_attr)
        else:
            hidden = self.conv1(x, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln1(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if edge_attr is not None:
            hidden = self.conv2(hidden, edge_index, edge_attr)
        else:
            hidden = self.conv2(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        hidden = self.ln2(hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden


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

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 gnn='gcn',
                 pooling='add',
                 edge_dim=None,
                 ):
        super(GNN_Net_Graph, self).__init__()
        max_depth = 3
        self.decoder = InnerProductDecoder()
        self.do_print = True
        print(f"dropout: {dropout}")
        self.dropout = dropout
        # Embedding (pre) layer
        self.bns = ModuleList()
        self.bns.append(BatchNorm(hidden))
        self.bns.append(BatchNorm(hidden))
        #self.bns.append(BatchNorm(hidden))
        # self.node_encoder = NodeEncoder(in_channels, hidden, edge_dim, dropout)
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
            self.gnn = GIN_Res_Net(in_channels=hidden,
                                   out_channels=hidden,
                                   hidden=hidden,
                                   max_depth=max_depth,
                                   dropout=dropout)
            self.private_net = GIN_Res_Net(in_channels=hidden,
                                           out_channels=hidden,
                                           hidden=hidden,
                                           max_depth=max_depth,
                                           dropout=dropout)


        elif gnn == 'gin_res':
            self.gnn = GIN_Res_Net(in_channels=hidden,
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

        self.linear1 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        #self.linear2 = Sequential(Linear(hidden, hidden//2), torch.nn.ReLU())
        #self.linear_resize_private = Linear(hidden, self.max_nodes)
        #self.linear_resize_shared = Linear(hidden, self.max_nodes)
        # self.layer_norm = torch.nn.LayerNorm(hidden)
        # self.layer_norm2 = torch.nn.LayerNorm(hidden)

        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')
        # x = self.node_encoder(x, edge_index, edge_attr)
        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)
        x = self.bns[0](x)
        # x = self.encoder2(x)
        # x = self.bns[1](x)
        # x = self.layer_norm(x)
        out_enc = self.gnn((x, edge_index))
        out_enc_pr = self.private_net((x, edge_index))
        # predicted adjacency matrix
        #out_enc_pr = self.linear_resize_private(out_enc_pr)
        #out_enc = self.linear_resize_shared(out_enc)
        sizes = degree(batch, dtype=torch.long).tolist()

        unbatched_enc_pr = out_enc_pr.split(sizes, 0)
        unbatched_enc = out_enc.split(sizes, 0)
        unbatched_edge_index = unbatch_edge_index(edge_index, batch)
        rec_loss = None
        for i in range(len(unbatched_enc)):
            #A_pred = dot_product_decode(unbatched_enc_pr[i] + unbatched_enc[i])
            if rec_loss is None:
                rec_loss = self.recon_loss(unbatched_enc_pr[i] + unbatched_enc[i], unbatched_enc_pr[i].size(dim=0),
                                            unbatched_edge_index[i], neg_edge_index=None, )
            else:
                rec_loss += self.recon_loss(unbatched_enc_pr[i] + unbatched_enc[i], unbatched_enc_pr[i].size(dim=0), unbatched_edge_index[i], neg_edge_index=None, )

        rec_loss = rec_loss/len(unbatched_enc)

        """
        print(out_enc.size())
        A_pred = dot_product_decode(out_enc + out_enc_pr)
        print(A_pred.size())
        if self.training:
            rec_loss = self.recon_loss(A_pred, x.size(dim=1), edge_index, neg_edge_index=None, )
        else:
            rec_loss = None

        """


        out_enc_pooled = self.pooling(out_enc, batch)
        # out_enc_pr_pooled = self.pooling(out_enc_pr, batch)

        x = out_enc_pooled  # + out_enc_pr_pooled
        #if self.training and self.do_print:
        #    print(f"shared encoder: {out_enc_pooled}")
        #    print(f"private encoder: {out_enc_pr_pooled}")
        #    self.do_print = False
        if not self.training:
            self.do_print = True

        # pr_x = self.private_net((x, edge_index))
        # x = x + enc
        # pr_x = self.pooling(x, batch)
        # pr_x = self.linear2(pr_x)

        x = self.linear1(x)
        # x = torch.cat((x, pr_x), 1)

        # x = self.linear3(x)
        x = self.bns[1](x)
        # x = self.layer_norm2(x)
        x = F.dropout(x, self.dropout, training=self.training)

        #x = self.shared_linear2(x)
        # x = self.bns[3](x)
        # x = F.dropout(x, self.dropout, training=self.training)
        #x = self.linear2(x)
        # x = self.linear3(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x, out_enc, out_enc_pr, rec_loss

    def recon_loss(self, z, num_nodes, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            try:
                neg_edge_index = negative_sampling(pos_edge_index, num_nodes=num_nodes)
            except:
                pass
        if neg_edge_index is not None:

            neg_loss = -torch.log(1 -
                                  self.decoder(z, neg_edge_index, sigmoid=True) +
                                  EPS).mean()
        else:
            neg_loss = 0

        return pos_loss + neg_loss


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return torch.nn.Parameter(initial)


class InnerProductDecoder(torch.nn.Module):
    """The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self):
        super().__init__()
    def forward(self, z, edge_index, sigmoid=True, neg_print =False):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)