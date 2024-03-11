import torch
from packaging.version import Version
from torch import nn
from torch.nn import functional as F


class InvConvNear(nn.Module):
    """Invertible Convolution with input splitting as in GlowTTS paper.
    https://arxiv.org/pdf/1811.00002.pdf

    Args:
        channels (int): input and output channels.
        num_splits (int): number of splits, also H and W of conv layer.
        no_jacobian (bool): enable/disable jacobian computations.

    Note:
        Split the input into groups of size self.num_splits and
        perform 1x1 convolution separately. Cast 1x1 conv operation
        to 2d by reshaping the input for efficiency.
    """

    def __init__(self, channels, num_splits=4, no_jacobian=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        assert num_splits % 2 == 0
        self.channels = channels
        self.num_splits = num_splits
        self.no_jacobian = no_jacobian
        self.weight_inv = None

        if Version(torch.__version__) < Version("1.9"):
            w_init = torch.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_())[0]
        else:
            w_init = torch.linalg.qr(torch.FloatTensor(self.num_splits, self.num_splits).normal_(), "complete")[0]

        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
        """
        b, c, t = x.size()
        assert c % self.num_splits == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.num_splits, self.num_splits // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.num_splits, c // self.num_splits, t)

        if reverse:
            if self.weight_inv is not None:
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.num_splits) * x_len  # [b]

        weight = weight.view(self.num_splits, self.num_splits, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.num_splits // 2, c // self.num_splits, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
        self.weight_inv = nn.Parameter(weight_inv, requires_grad=False)