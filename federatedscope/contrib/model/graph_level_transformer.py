import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
#from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import TransformerConv, LayerNorm, GCNConv
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, \
    global_max_pool


#from federatedscope.gfl.model.gcn import GCN_Net
#from federatedscope.gfl.model.sage import SAGE_Net
#from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
#from federatedscope.gfl.model.gpr import GPR_Net

EMD_DIM = 200

class AtomEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i in range(hidden):
            emb = torch.nn.Embedding(EMD_DIM, hidden)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i+1](x[:, i])
        return x_embedding

class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, edge_dim, dropout):
        super(NodeEncoder, self).__init__()
        self.dropout = dropout
        edge_dim = None if edge_dim == 0 else edge_dim
        self.conv1 = TransformerConv(in_channels, hidden, edge_dim=edge_dim)
        self.ln1 = LayerNorm(hidden)
        self.conv2 = TransformerConv(hidden, hidden, edge_dim=edge_dim)
        self.ln2 = LayerNorm(hidden)

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


class GNN_Net_Graph_Transformer(torch.nn.Module):
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
                 dropout=0.5,
                 gnn='gcn',
                 pooling='add',
                 edge_dim=None,
                 learning_type = ""):
        super(GNN_Net_Graph_Transformer, self).__init__()
        self.dropout = dropout
        # Embedding (pre) layer
        self.dropout = dropout
        edge_dim = None if edge_dim == 0 else edge_dim
        self.learning_type = learning_type
        self.out_channels = out_channels





        self.node_encoder = NodeEncoder(in_channels, hidden, edge_dim, dropout)
        # self.encoder_atom = AtomEncoder(in_channels, hidden)
        # self.encoder = Linear(in_channels, hidden)


        self.gnn_str = gnn
        # GNN layer

        if gnn == 'gin' or gnn=='transformer':
            self.gnn = GIN_Net(in_channels=hidden,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)


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
        #self.conv_gcn = GCNConv(hidden, hidden)
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)





    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        #elif isinstance(data, tuple):
        #    x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.node_encoder(x, edge_index, edge_attr)
        x = self.gnn((x, edge_index))
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x

    """
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

        x = self.gnn((x, edge_index))
        #x = self.gnn2((x, edge_index))
        #x = self.conv_gcn(x, edge_index)
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x
    """
    #def load_state_dict(self, state_dict, strict: bool = True):
    #    state_dict
    #    state_dict[self.name_reserve] = getattr(self, self.name_reserve)
    #    super().load_state_dict(state_dict, strict)

