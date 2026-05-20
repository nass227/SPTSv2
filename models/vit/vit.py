# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from util.misc_sptsv2 import NestedTensor


class ViTPatchEmbedder(nn.Module):
    """
    Backbone ViT pré-entraîné (vit_small_patch16_224 sur ImageNet).
    Produit (features, pos) compatibles avec SPTSv2 :
        features : liste d'un NestedTensor  (src, mask) en 2D spatiale
        pos      : [None] — positional encoding déjà dans le ViT
    """

    def __init__(
        self,
        img_size: int  = 640,
        patch_size: int = 16,
        embed_dim: int  = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.img_size  = 224   # ViT pré-entraîné attend 224×224
        self.embed_dim = embed_dim

        # ── ViT-Small pré-entraîné sur ImageNet ─────────────────────────
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained  = pretrained,
            num_classes = 0,        # supprimer la tête de classification
        )

        vit_dim = self.vit.embed_dim  # 384 pour vit_small

        # ── projection 384 → embed_dim (256) ────────────────────────────
        self.proj = nn.Linear(vit_dim, embed_dim)

        self.num_channels = embed_dim  # lu par SPTSv2 → input_proj
        self.grid_size    = (14, 14)   # 224/16 = 14 patches par côté

    def forward(self, samples: NestedTensor):
        x       = samples.tensors  # (B, 3, H, W)
        mask_in = samples.mask     # (B, H, W)

        B = x.shape[0]
        H_p, W_p = self.grid_size  # (14, 14)

        # ── redimensionner à 224×224 (requis par ViT pré-entraîné) ──────
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False,
            )

        # ── extraire les features du ViT ─────────────────────────────────
        # forward_features retourne (B, N+1, vit_dim)
        # N+1 = 196 patches + 1 token CLS
        tokens = self.vit.forward_features(x)  # (B, 197, 384)
        tokens = tokens[:, 1:, :]              # supprimer CLS → (B, 196, 384)
        tokens = self.proj(tokens)             # (B, 196, 256)

        # ── reshape → feature map 2D ─────────────────────────────────────
        src = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, H_p, W_p)
        # (B, 256, 14, 14)

        # ── masque à la résolution des patches ───────────────────────────
        mask = F.interpolate(
            mask_in.unsqueeze(1).float(),
            size=(H_p, W_p),
            mode='nearest',
        ).squeeze(1).bool()  # (B, 14, 14)

        pos = torch.zeros_like(src)

        feature = NestedTensor(src, mask)
        return [feature], [pos]   # pos=None → with_pos_embed ignoré


def build_vit_backbone(args) -> ViTPatchEmbedder:
    """Factory function appelée depuis sptsv2.py."""
    pretrained = getattr(args, 'pretrained_vit', True)
    return ViTPatchEmbedder(
        img_size   = getattr(args, 'img_size',   640),
        patch_size = getattr(args, 'patch_size',  16),
        embed_dim  = getattr(args, 'embed_dim',  256),
        pretrained = pretrained,
    )