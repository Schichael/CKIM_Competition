import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.nn import BatchNorm1d, Identity


class MLPResNet(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self,
                 channel_list,
                 dropout=0.,
                 batch_norm=True,
                 relu_first=False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first
        self.in_channels = channel_list[0]
        self.local_out = None
        self.global_out = None

        self.linears = ModuleList()

        self.norms = ModuleList()
        for in_channel, out_channel in zip(channel_list[:-1],
                                           channel_list[1:]):
            self.local_linear = Linear(in_channel, out_channel)
            self.local_norm = BatchNorm1d(out_channel) if batch_norm else Identity()
            self.linears.append(Linear(in_channel, out_channel))
            self.norms.append(
                BatchNorm1d(out_channel) if batch_norm else Identity())

    def forward(self, x):
        x_local = self.local_linear(x)
        if self.relu_first:
            x_local = F.relu(x_local)
        x_local = self.local_norm(x_local)
        self.local_out = x_local


        x_global = self.linears[0](x)

        for layer, norm in zip(self.linears[1:], self.norms[:-1]):
            if self.relu_first:
                x_global = F.relu(x_global)
            x_global = norm(x_global)
            if not self.relu_first:
                x_global = F.relu(x_global)
            x_global = F.dropout(x_global, p=self.dropout, training=self.training)
            x_global = layer.forward(x_global)
        x = x_global + x_local
        self.global_out = x_global
        return x


