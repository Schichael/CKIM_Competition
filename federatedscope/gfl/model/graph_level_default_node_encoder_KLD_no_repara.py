import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, Sequential, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net

EMD_DIM = 200
# graph_level_default_node_encoder_KLD_no_repara

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

class VAE_Decoder(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,):
        super(VAE_Decoder, self).__init__()
        self.lin1 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.bn1 = BatchNorm1d(hidden)
        self.lin2 = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.bn2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, out_channels)

    def forward(self, h):
        out = self.lin1(h)
        out = self.bn1(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.clf(out)
        return out



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
        self.dropout = dropout
        self.hidden= hidden
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
            self.gnn = GIN_Net(in_channels=hidden,
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
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)
        self.vae_decoder = VAE_Decoder(out_channels, in_channels, hidden)

    #def kld_loss(self, mu, log_var):
    #    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # In https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        # the number of minibatch samples is multiplied with the loss
    #    return kld_loss

    def kld_loss(self, x):
        mu = torch.mean(x, dim=-2)
        std = torch.std(x, dim=-2)
        log_var = torch.log(std) * 2
        kld_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())
        # In https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        # the number of minibatch samples is multiplied with the loss
        return kld_loss

    def reparametrize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            vector_size = log_var.size()
            eps = Variable(torch.FloatTensor(vector_size).normal_()).to('cuda:0')
            return eps.mul(std).add_(mu)
            # return eps * std + mu
        else:
            return mu
            #return torch.zeros(mu.size()).to('cuda:0')

    def vae_loss(self, mu, log_var, x_orig, x_decoded):
        kld_loss = self.kld_loss(mu, log_var)
        # recon_loss = F.mse_loss(x_decoded, x_orig)
        loss = kld_loss
        return loss

    def forward(self, data):
        if isinstance(data, Batch):
            x_in, edge_index, batch = data.x, data.edge_index, data.batch
        elif isinstance(data, tuple):
            x_in, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        if x_in.dtype == torch.int64:
            x = self.encoder_atom(x_in)
        else:
            x = self.encoder(x_in)

        mu = torch.mean(x, dim=-2)
        std = torch.std(x, dim=-2)

        kld_loss = self.kld_loss(x)
        x = self.gnn((x, edge_index))
        x = self.pooling(x, batch)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x, kld_loss, mu, std
