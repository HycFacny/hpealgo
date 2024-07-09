from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from einops import repeat
from einops import rearrange
from timm.models.layers.weight_init import trunc_normal_

from models.network_modules import Bottleneck
from models.transformer_base import Transformer
from models.transformer_base import MLPHead
from utils.print_functions import print_inter_debug_info



class TokenPoseBase(nn.Module):
    """
        tokenpose base
        left resnet fusion methods, transformer layers, and mlp head only
    """

    def __init__(
        self,
        feature_size,
        patch_size,
        num_keypoints,
        dim,
        apply_init=False,
        heatmap_size=[48, 64],
        channels=3,
        embedding_dropout=0.,
        patch_scale=1,
        pos_embedding_type='learnable'
    ):
        super().__init__()

        assert isinstance(feature_size, list) and isinstance(patch_size, list), \
            'feature_size and patch_size must be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0, \
            'feature dimensions must be divisible by patch dimensions'
        assert pos_embedding_type in ['sine', 'none', 'learnable', 'sine-full'], \
            'position embedding type is not in [\'sine\', \'none\', \'learnable\', \'sine-full\']'

        self.num_keypoints = num_keypoints
        self.num_patches = (feature_size[0] // (patch_scale * patch_size[0])) * \
                           (feature_size[1] // (patch_scale * patch_size[1]))
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size[0] * patch_size[1]
        self.in_channels = 64
        self.heatmap_size = heatmap_size
        self.pos_embedding_type = pos_embedding_type
        self.all_attention = pos_embedding_type == 'sine-full'

        self.pos_embedding_w, self.pos_embedding_h = \
            feature_size[0] // (patch_scale * patch_size[0]), feature_size[1] // (patch_scale * patch_size[1])
        self._make_position_embedding(dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.dropout = nn.Dropout(embedding_dropout)
        
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        self.to_keypoint_token = nn.Identity()

        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init: self.apply(self._init_weights)


    def forward():
        raise NotImplementedError

    def _make_position_embedding(self, d_model, pos_embedding_type):
        if pos_embedding_type == 'none':
            self.pos_embedding = None
            print('=> Without any position embedding.')
            return
        
        with torch.no_grad():
            if pos_embedding_type == 'learnable':
                # concat feature maps token and keypoint token together
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, self.num_patches + self.num_keypoints, d_model),
                    requires_grad=True
                )
                trunc_normal_(self.pos_embedding, std=.02)
                print('=> Add learnable position embedding')
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False
                )
                print('=> Add sine position embedding')

    def _make_sine_position_embedding(self, d_model, temperature=int(1e4), scale=2 * np.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w

        w, h = self.pos_embedding_w, self.pos_embedding_h
        area_map = torch.ones(1, h, w)          # [b, h, w]
        y_embed = area_map.cumsum(1, dtype=torch.float32)
        x_embed = area_map.cumsum(2, dtype=torch.float32)
        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        dim_temp = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_temp = temperature ** (2 * (dim_temp // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_temp
        pos_y = y_embed[:, :, :, None] / dim_temp

        pos_x = torch.stack((
                pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()
            ), dim=4
        ).flatten(3)
        pos_y = torch.stack((
                pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()
            ), dim=4
        ).flatten(3)

        pos = torch.cat( (pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2)        # [1, d_model, h x w]
        pos = pos.permute(0, 2, 1)  # [1, h x w, d_model]
        
        return pos

    def _init_weights(self, method):
        print('Initializating......')
        if isinstance(method, nn.Linear):
            trunc_normal_(method.weight, std=.02)
            nn.init.constant_(method.bias, 0.)
        elif isinstance(method, nn.LayerNorm):
            nn.init.constant_(method.weight, 1.)
            nn.init.constant_(method.bias, 0.)


class TokenPose_L_Base(TokenPoseBase):

    def __init__(
        self,
        *,
        feature_size,
        patch_size,
        num_keypoints,
        dim,
        depth,
        heads,
        mlp_dim,
        apply_init=False,
        apply_multi=True,
        hidden_heatmap_dim=64 * 6,
        heatmap_dim=48 * 64,
        heatmap_size=[48, 64],              # w, h
        channels=3,
        dropout=0.,
        embedding_dropout=0.,
        pos_embedding_type='learnable'
    ):
        super(TokenPose_L_Base, self).__init__(
            feature_size, patch_size, num_keypoints, dim, apply_init, heatmap_size, channels, embedding_dropout, 1, pos_embedding_type
        )
        
        # transformer
        self.transformer1 = Transformer(dim, depth, heads, mlp_dim, dropout, 
            num_keypoints=num_keypoints, all_attention=self.all_attention, scale_with_head=True)
        self.transformer2 = Transformer(dim, depth, heads, mlp_dim, dropout, 
            num_keypoints=num_keypoints, all_attention=self.all_attention, scale_with_head=True)
        self.transformer3 = Transformer(dim, depth, heads, mlp_dim, dropout, 
            num_keypoints=num_keypoints, all_attention=self.all_attention, scale_with_head=True)
        
        # mlp head
        self.mlp_head = nn.Sequential(
            MLPHead(dim * 3, hidden_heatmap_dim),
            MLPHead(hidden_heatmap_dim, heatmap_dim)
        ) if dim * 3 <= hidden_heatmap_dim * 0.5 and apply_multi else MLPHead(dim * 3, heatmap_dim)

    def forward(self, features, mask=None):
        # print(features.shape)
        p_h, p_w = self.patch_size[1], self.patch_size[0]

        # backbone feature: [b, c, h, w] ( b: batch_size, c: channels, h: featuremap_h, w: featuremap_w)
        # transform [b, c, h, w] to [b, (h' * w'), d_model], d_model = p_h * p_w * c
        out = rearrange(features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p_h, p2=p_w)
        # out: [b, patch_nums, embedding_dim]
        out = self.patch_to_embedding(out)

        b, n, _ = out.shape

        # keypoint tokens params: [batch_size, keypoint_nums, dim]
        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
        # print(self.pos_embedding.shape)
        # print(out.shape)
        if self.pos_embedding_type in ['sine', 'sine-full']:
            out += self.pos_embedding[:, :n]
            out = torch.cat( (keypoint_tokens, out), dim=1)
        else:
            out = torch.cat( (keypoint_tokens, out), dim=1)
            out += self.pos_embedding[:, :(n + self.num_keypoints)]
        
        out = self.dropout(out)

        # out: [batch_size, 256 + 17, 192]
        x1 = self.transformer1(out, mask, self.pos_embedding)
        x2 = self.transformer2(x1,  mask, self.pos_embedding)
        x3 = self.transformer3(x2,  mask, self.pos_embedding)

        x1_out = self.to_keypoint_token(x1[:, 0 : self.num_keypoints])
        x2_out = self.to_keypoint_token(x2[:, 0 : self.num_keypoints])
        x3_out = self.to_keypoint_token(x3[:, 0 : self.num_keypoints])

        out = torch.cat( (x1_out, x2_out, x3_out), dim=2 )
        out = self.mlp_head(out)

        out = rearrange(out, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[1], p2=self.heatmap_size[0])

        return out