# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, representation_method='concat', use_ln=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        self.representation_method = representation_method

        print('Using representation method ', self.representation_method)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln = nn.LayerNorm(normalized_shape=768, elementwise_affine=True)
            print('Using layer norm')
            
        if self.representation_method == 'concat':
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"] * 2 # concatenating so therefore the dim doubles
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        elif self.representation_method == 'global_pool':
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.use_ln: #if fine tuned VisionFM doesnt work in linear probing, it could be because of placement of this LN
            x = self.ln(x)
        
        if self.representation_method == 'concat':
            patches = x[:, 1:, :].mean(dim=1)
            cls = x[:, 0]
            concatenated = torch.concat((patches, cls), 1)            
            if self.use_ln:
                outcome = concatenated                
            else:
                outcome = self.fc_norm(concatenated)
        elif self.representation_method == 'global_pool':
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        # Clumped down a lot, original mae uses old timm, latest timm is too new
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

