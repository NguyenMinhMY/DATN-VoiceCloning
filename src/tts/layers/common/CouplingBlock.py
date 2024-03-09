import torch
from torch import nn

from src.tts.layers.common.WaveNet import WaveNet

class CouplingBlock(nn.Module):
    """Glow Affine Coupling block as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf

    ::

        x --> x0 -> conv1d -> wavenet -> conv1d --> t, s -> concat(s*x1 + t, x0) -> o
        '-> x1 - - - - - - - - - - - - - - - - - - - - - - - - - ^

    Args:
         in_channels (int): number of input tensor channels.
         hidden_channels (int): number of hidden channels.
         kernel_size (int): WaveNet filter kernel size.
         dilation_rate (int): rate to increase dilation by each layer in a decoder block.
         num_layers (int): number of WaveNet layers.
         c_in_channels (int): number of conditioning input channels.
         dropout_p (int): wavenet dropout rate.
         sigmoid_scale (bool): enable/disable sigmoid scaling for output scale.

    Note:
         It does not use the conditional inputs differently from WaveGlow.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        c_in_channels=0,
        dropout_p=0,
        sigmoid_scale=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.c_in_channels = c_in_channels
        self.dropout_p = dropout_p
        self.sigmoid_scale = sigmoid_scale
        # input layer
        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        # start = torch.nn.utils.parametrizations.weight_norm(start)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        # coupling layers
        self.wn = WaveNet(hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels, dropout_p)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, : self.in_channels // 2], x[:, self.in_channels // 2 :]

        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)

        z_0 = x_0
        t = out[:, : self.in_channels // 2, :]
        s = out[:, self.in_channels // 2 :, :]
        if self.sigmoid_scale:
            s = torch.log(1e-6 + torch.sigmoid(s + 2))

        if reverse:
            z_1 = (x_1 - t) * torch.exp(-s) * x_mask
            logdet = None
        else:
            z_1 = (t + torch.exp(s) * x_1) * x_mask
            logdet = torch.sum(s * x_mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()