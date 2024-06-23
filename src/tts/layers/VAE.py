import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from src.tts.layers.common.Convolution import ConvolutionModule

class VAEEncoder(nn.Module):
    def __init__(self, i_dim, hidden_dim, c_latent):
        super(VAEEncoder, self).__init__()
        self.pre_net = nn.Conv1d(i_dim, hidden_dim, kernel_size=1)
        self.nn = ConvolutionModule(channels=hidden_dim, kernel_size=31)
        self.out_proj = nn.Conv1d(hidden_dim, c_latent * 2, kernel_size=1)
        self.latent_channels = c_latent


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pre_net(x)
        x = x.transpose(1, 2)
        x = self.nn(x)
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        mu, logvar = torch.split(x, self.latent_channels, dim=1)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, c_latent, hidden_dim, o_dim):
        super(VAEDecoder, self).__init__()
        self.pre_net = nn.Conv1d(c_latent, hidden_dim, kernel_size=1)
        self.nn = ConvolutionModule(channels=hidden_dim, kernel_size=31)
        self.out_proj = nn.Conv1d(hidden_dim, o_dim, kernel_size=1)

    def forward(self, x):
        x = self.pre_net(x)
        x = x.transpose(1, 2)
        x = self.nn(x)
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        return x.transpose(1, 2)

class VAE(nn.Module):
    def __init__(self, i_dim=384, hidden_dim=256, c_latent=64):
        super(VAE, self).__init__()
        self.latent_channels = c_latent
        self.encoder = VAEEncoder(i_dim, hidden_dim, c_latent)
        self.decoder = VAEDecoder(c_latent, hidden_dim, i_dim)
        self.prior_dist = dist.Normal(0, 1)

    def forward(self, x):
        mu_q, logvar_q = self.encoder(x)
        q_dist = dist.Normal(mu_q, torch.exp(0.5 * logvar_q))

        sampled_latent = self.reparameterize(mu_q, logvar_q)
        reconstructed_features = self.decoder(sampled_latent)

        kl_loss = dist.kl_divergence(q_dist, self.prior_dist).mean()
        vae_reconstruction_loss = nn.MSELoss()(reconstructed_features, x) 

        return kl_loss, vae_reconstruction_loss, reconstructed_features       

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

