from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

from models.network_modules import Residual


MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


class PreNorm(nn.Module):
    def __init__(self, dim, net, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.net = net
    
    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim / heads) ** -.5 if scale_with_head else dim ** -.5        # zoom factor, e.g. \sqrt{d}

        # concat q, k, v to one linear layer
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self.num_keypoints = num_keypoints
    
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        # split to_qkv output into Q, K, V tuple
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print(b, n, _, h)
        # get q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # QxK^T
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_( ~mask, mask_value )
            del mask

        attention = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Transformer(nn.Module):

    def __init__(
        self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attention=False, scale_with_head=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attention = all_attention
        self.num_keypoints = num_keypoints

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual( PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head)) ),
                Residual( PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)) )
            ]))
    
    def forward(self, x, mask=None, pos=None):
        for idx, (attention, feedforward) in enumerate(self.layers):
            # print(idx, attention, feedforward)
            if idx > 0 and self.all_attention:
                x[:, self.num_keypoints:] += pos
            x = attention(x, mask=mask)
            x = feedforward(x)
        return x


class MLPHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.net(x)
