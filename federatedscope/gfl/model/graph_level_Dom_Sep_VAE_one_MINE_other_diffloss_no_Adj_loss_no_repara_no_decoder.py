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

# graph_level_Dom_Sep_VAE_one_MINE_other_diffloss_no_Adj_loss_no_repara_no_decoder

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

class VAE_Decoder(torch.nn.Module):
    def __init__(self,
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

class DiffLoss(torch.nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss



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
        self.hidden = hidden

        self.rho=rho
        print(f"rho: {rho}")
        if edge_dim is None or edge_dim == 0:
            edge_dim = 1
        super(GNN_Net_Graph, self).__init__()
        self.diff_loss = DiffLoss()
        self.dropout = dropout
        # Embedding (pre) layer
        self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        self.cos_loss = torch.nn.CosineEmbeddingLoss()
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

            self.fixed_gnn = GIN_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)

            for param in self.fixed_gnn.named_parameters():
                param[1].requires_grad = False

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

        # Output layer
        self.global_linear_out1 = Linear(hidden, hidden)
        #self.linear_out2_glob = Sequential(Linear(hidden, 64))
        #self.bn_linear0_glob = BatchNorm1d(hidden * max_depth)
        #self.bn_linear1_glob = BatchNorm1d(hidden)
        #self.bn_linear2_glob = BatchNorm1d(64)
        self.local_linear_out1 = Linear(hidden, hidden)
        #self.linear_out2_loc = Sequential(Linear(hidden, 64))
        self.bn_linear0_loc = BatchNorm1d(hidden * max_depth)
        self.bn_linear1_loc = BatchNorm1d(hidden)
        self.bn_after_summation = BatchNorm1d(hidden)
        #self.bn_linear2_loc = BatchNorm1d(64)


        # local
        self.linear_out2 = Sequential(Linear(hidden, hidden))
        self.bn_linear2 = BatchNorm1d(hidden)
        self.clf = Linear(hidden, out_channels)
        self.emb = Linear(edge_dim, hidden)
        #torch.nn.init.xavier_normal_(self.emb.weight.data)

    def kld_loss(self, x):
        mu = torch.mean(x, dim=-2)
        std = torch.std(x, dim=-2)
        log_var = torch.log(std) * 2
        kld_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=0)
        # In https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        # the number of minibatch samples is multiplied with the loss
        return kld_loss

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def node_recon_loss(self, x_orig, x_decoded):
        #cosine embedding loss: 1-cos(x1, x2). The 1 defines this loss function.
        y = torch.ones(x_decoded.size(0)).to('cuda:0')
        recon_loss = self.cos_loss(x_decoded, x_orig, y)
        return recon_loss

    def recon_loss_adj(self, z, num_nodes, pos_edge_index, neg_edge_index=None):
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

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        if x.dtype == torch.int64:
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)

        kld_loss = self.kld_loss(x)

        x_local_enc = self.local_gnn((x, edge_index))

        x_global_enc = self.global_gnn((x, edge_index))
        x_fixed_enc = self.fixed_gnn((x, edge_index))
        x_global = self.pooling(x_global_enc, batch)

        x_global = self.global_linear_out1(x_global).relu()


        x_local = self.pooling(x_local_enc, batch)
        x_local = self.local_linear_out1(x_local).relu()

        diff_local_global = self.diff_loss(x_local_enc, x_global_enc)
        mi_global_fixed = self.mine(x_global_enc, x_fixed_enc)

        x = x_local + x_global

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.clf(x)

        return x, kld_loss, 0, diff_local_global, mi_global_fixed







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