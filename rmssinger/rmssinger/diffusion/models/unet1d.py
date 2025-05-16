# portaspeech-like AE model, which can be viewed as a kind of Unet
import numpy as np
import torch
from torch import nn
import math
from modules.commons.conv import ConditionalConvBlocks
from modules.commons.wavenet import WN
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FAEEncoder(nn.Module):
    def __init__(self, c_in, hidden_size, c_latent, kernel_size,
                 n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        if np.prod(strides) == 1:
            self.pre_net = nn.Conv1d(c_in, hidden_size, kernel_size=1)
        else:
            self.pre_net = nn.Sequential(*[
                nn.Conv1d(c_in, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2)
                if i == 0 else
                nn.Conv1d(hidden_size, hidden_size, kernel_size=s * 2, stride=s, padding=s // 2)
                for i, s in enumerate(strides)
            ])
        if nn_type == 'wn':
            self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
        elif nn_type == 'conv':
            self.nn = ConditionalConvBlocks(
                hidden_size, c_cond, hidden_size, None, kernel_size,
                layers_in_block=2, is_BTC=False, num_layers=n_layers)

        self.out_proj = nn.Conv1d(hidden_size, c_latent, 1)
        self.latent_channels = c_latent

    def forward(self, x, nonpadding, cond, temb):
        x = self.pre_net(x)
        x = x + temb[:, :, None]
        nonpadding = nonpadding[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
        x = x * nonpadding
        x = self.nn(x, nonpadding=nonpadding, cond=cond) * nonpadding
        x = x + temb[:, :, None]
        return x


class FAEDecoder(nn.Module):
    def __init__(self, c_latent, hidden_size, out_channels, kernel_size,
                 n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
        super().__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.pre_net = nn.Sequential(*[
            nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=s, stride=s)
            if i == 0 else
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=s, stride=s)
            for i, s in enumerate(strides)
        ])
        if nn_type == 'wn':
            self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
        elif nn_type == 'conv':
            self.nn = ConditionalConvBlocks(
                hidden_size, c_cond, hidden_size, [1] * n_layers, kernel_size,
                layers_in_block=2, is_BTC=False)
        self.out_proj = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, x, nonpadding, cond, temb):
        x = self.pre_net(x)
        x = x + temb[:, :, None]
        x = x * nonpadding
        x = self.nn(x, nonpadding=nonpadding, cond=cond) * nonpadding
        x = self.out_proj(x)
        return x


class FAE(nn.Module):
    def __init__(self,
                 c_in_out, hidden_size, c_latent,
                 kernel_size, enc_n_layers, dec_n_layers, c_cond, strides,
                 use_prior_flow,encoder_type='wn', decoder_type='wn'):
        super(FAE, self).__init__()
        self.strides = strides
        self.hidden_size = hidden_size
        self.latent_size = c_latent
        self.use_prior_flow = use_prior_flow
        if np.prod(strides) == 1:
            self.g_pre_net = nn.Conv1d(c_cond, c_cond, kernel_size=1)
        else:
            self.g_pre_net = nn.Sequential(*[
                nn.Conv1d(c_cond, c_cond, kernel_size=s * 2, stride=s, padding=s // 2)
                for i, s in enumerate(strides)
            ])
        self.encoder = FAEEncoder(c_in_out, hidden_size, c_latent, kernel_size,
                                   enc_n_layers, c_cond, strides=strides, nn_type=encoder_type)
        self.decoder = FAEDecoder(c_latent, hidden_size, c_in_out, kernel_size,
                                   dec_n_layers, c_cond, strides=strides, nn_type=decoder_type)
        self.diffusion_embedding = SinusoidalPosEmb(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4),
            Mish(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x=None, timestep=None, cond=None, nonpadding=None):
        """

        :param x: [B, C_in_out, T]
        :param nonpadding: [B, 1, T]
        :param cond: [B, C_g, T]
        :return:
        """
        timestep = self.diffusion_embedding(timestep)
        timestep = self.mlp(timestep)
        if nonpadding is None:
            nonpadding = 1
        cond_sqz = self.g_pre_net(cond)
        x = self.encoder(x, nonpadding, cond_sqz, timestep)
        out = self.decoder(x, nonpadding, cond, timestep)
        return out
       