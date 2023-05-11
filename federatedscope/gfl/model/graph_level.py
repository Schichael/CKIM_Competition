import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, Sequential, Parameter
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net

EMD_DIM = 200
#graph_level_cross_stitch

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
        dropout (float): dropout pdefaultrobability.
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
        self.dropout = dropout
        self.cos_loss = torch.nn.CosineEmbeddingLoss()
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
            self.global_gnn_1 = GIN_Net(in_channels=hidden,
                                        out_channels=hidden,
                                        hidden=hidden,
                                        max_depth=1,
                                        dropout=dropout)
            self.global_gnn_2 = GIN_Net(in_channels=hidden,
                                        out_channels=hidden,
                                        hidden=hidden,
                                        max_depth=1,
                                        dropout=dropout)

            self.local_gnn_1 = GIN_Net(in_channels=hidden,
                                        out_channels=hidden,
                                        hidden=hidden,
                                        max_depth=1,
                                        dropout=dropout)
            self.local_gnn_2 = GIN_Net(in_channels=hidden,
                                        out_channels=hidden,
                                        hidden=hidden,
                                        max_depth=1,
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
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.local_linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

        # cross_stitch_units
        self.local_alpha_1 = Parameter(torch.Tensor([[0.9, 0.1], [0.9, 0.1]]),
                                      requires_grad=False)
        self.local_alpha_2 = Parameter(torch.Tensor([[0.9, 0.1], [0.9, 0.1]]),
                                      requires_grad=False)
        self.local_alpha_3 = Parameter(torch.Tensor([[0.9, 0.1], [0.9, 0.1]]),
                                      requires_grad=False)


    def cosine_diff_loss(self, x1, x2):
        # cosine embedding loss: 1-cos(x1, x2). The 1 defines this loss function.
        y = torch.ones(x1.size(0)).to('cuda:0')
        y = -y
        diff_loss = self.cos_loss(x1, x2, y)
        return diff_loss


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

        x_global_1 = self.global_gnn_1((x, edge_index))

        x_local_1 = self.local_gnn_1((x, edge_index))

        diff_1 = self.cosine_diff_loss(x_local_1, x_global_1)

        x_global_1_cs = self.local_alpha_1[0][0]*x_global_1 + self.local_alpha_1[0][
            1]*x_local_1

        x_local_1_cs = self.local_alpha_1[1][0] * x_global_1 + \
                         self.local_alpha_1[1][1] * x_local_1


        x_global_1 = F.relu(F.dropout(x_global_1_cs, p=self.dropout, training=self.training))
        x_local_1 = F.relu(F.dropout(x_local_1_cs, p=self.dropout, training=self.training))

        x_global_2 = self.global_gnn_2((x_global_1, edge_index))
        x_local_2 = self.local_gnn_2((x_local_1, edge_index))

        x_global_2 = self.pooling(x_global_2, batch)

        x_local_2 = self.pooling(x_local_2, batch)

        diff_2 = self.cosine_diff_loss(x_local_2, x_global_2)

        x_global_2_cs = self.local_alpha_2[0][0] * x_global_2 + self.local_alpha_2[0][
            1] * x_local_2

        x_local_2_cs = self.local_alpha_2[1][0] * x_global_2 + \
                    self.local_alpha_2[1][1] * x_local_2



        x_global_3 = self.linear(x_global_2_cs)
        x_local_3 = self.local_linear(x_local_2_cs)

        diff_3 = self.cosine_diff_loss(x_local_3, x_global_3)

        x_global_3_cs = self.local_alpha_3[0][0] * x_global_3 + self.local_alpha_3[0][
            1] * x_local_3

        x = F.dropout(x_global_3_cs, self.dropout, training=self.training)
        x = self.clf(x)
        return x, self.local_alpha_1, self.local_alpha_2, self.local_alpha_3, diff_1,\
            diff_2, diff_3, x_local_1, x_local_2, x_local_3, x_global_1, \
            x_global_2, x_global_3, x_global_1_cs, x_global_2_cs, x_global_3_cs, x_local_1_cs, x_local_2_cs, x_global_3_cs # only
        # use diff_3
