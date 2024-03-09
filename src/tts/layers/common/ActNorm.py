import torch
from torch import nn

class ActNorm(nn.Module):
    """Activation Normalization bijector as an alternative to Batch Norm. It computes
    mean and std from a sample data in advance and it uses these values
    for normalization at training.

    Args:
        channels (int): input channels.
        ddi (False): data depended initialization flag.

    Shapes:
        - inputs: (B, C, T)
        - outputs: (B, C, T)
    """

    def __init__(self, channels, ddi=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m**2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)