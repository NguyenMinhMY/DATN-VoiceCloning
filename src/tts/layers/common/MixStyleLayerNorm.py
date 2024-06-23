import random

import torch
import torch.nn as nn


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256):
        """
        Args:
            p (float): probability of using MixStyle.
            alpha (float): parameter of the Beta distribution.
            eps (float): scaling parameter to avoid numerical issues.
            mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            hidden_size,
            2 * hidden_size,  # For both b (bias) g (gain)
        )

    def forward(self, x, spk_embed, is_inference=False):
        if not is_inference and random.random() > self.p:
            return x

        B = x.size(0)

        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(
            x, dim=-1, keepdim=True
        )
        x_normed = (x - mu) / (sig + 1e-6)  # [B, T, H_m]

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        # Get Bias and Gain
        mu1, sig1 = torch.split(
            self.affine_layer(spk_embed), self.hidden_size, dim=-1
        )  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # MixStyle
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]

        if not is_inference:
            mu_mix = mu1 * lmda + mu2 * (1 - lmda)
            sig_mix = sig1 * lmda + sig2 * (1 - lmda)
        else:
            mu_mix = mu1
            sig_mix = sig1

        # Perform Scailing and Shifting
        return sig_mix * x_normed + mu_mix  # [B, T, H_m]


class MixStyle2(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256, spk_embed_dim=64):
        """
        Args:
            p (float): probability of using MixStyle.
            alpha (float): parameter of the Beta distribution.
            eps (float): scaling parameter to avoid numerical issues.
            mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.spk_embed_dim = spk_embed_dim
        self.affine_layer = nn.Sequential(
            nn.Linear(self.spk_embed_dim, self.spk_embed_dim),
            nn.Tanh(),
            nn.Linear(self.spk_embed_dim, self.hidden_size),
            nn.Tanh(),
            LinearNorm(hidden_size, 2 * hidden_size),  # For both b (bias) g (gain)
        )

    def forward(self, x, spk_embed, is_inference=False):
        if not is_inference and random.random() > self.p:
            return x

        B = x.size(0)

        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(
            x, dim=-1, keepdim=True
        )
        x_normed = (x - mu) / (sig + 1e-6)  # [B, T, H_m]

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        # Get Bias and Gain
        mu1, sig1 = torch.split(
            self.affine_layer(spk_embed), self.hidden_size, dim=-1
        )  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]

        # MixStyle
        perm = torch.randperm(B)
        mu2, sig2 = mu1[perm], sig1[perm]

        if not is_inference:
            mu_mix = mu1 * lmda + mu2 * (1 - lmda)
            sig_mix = sig1 * lmda + sig2 * (1 - lmda)
        else:
            mu_mix = mu1
            sig_mix = sig1

        # Perform Scailing and Shifting
        return sig_mix * x_normed + mu_mix  # [B, T, H_m]
