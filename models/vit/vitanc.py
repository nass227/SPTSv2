# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from util.misc_sptsv2 import NestedTensor


class ViTPatchEmbedder(nn.Module):
    """
    Remplace le backbone CNN par un patch embedder ViT.
    Produit (features, pos) compatibles avec SPTSv2 :
        features : liste d'un NestedTensor  (src, mask) en 2D spatiale
        pos      : liste d'un tenseur zeros (B, embed_dim, H_p, W_p)
    Le positional encoding est déjà fusionné dans les tokens.
    """

    def __init__(
        self,
        img_size: int = 640,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size     = img_size
        self.patch_embed  = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches  = self.patch_embed.num_patches
        self.num_channels = embed_dim                    # lu par SPTSv2 → input_proj
        self.grid_size    = self.patch_embed.grid_size   # (H_p, W_p)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, samples: NestedTensor):
        x       = samples.tensors  # (B, 3, H, W)
        mask_in = samples.mask     # (B, H, W) – 1 sur pixels paddés

        B = x.shape[0]
        H_p, W_p = self.grid_size

        # ── redimensionner l'image à img_size si nécessaire ─────────────
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        # ── patch embedding + positional encoding ───────────────────────
        tokens = self.patch_embed(x)        # (B, N, embed_dim)
        tokens = tokens + self.pos_embed

        # ── reshape → feature map 2D ────────────────────────────────────
        src = tokens.permute(0, 2, 1).reshape(B, -1, H_p, W_p)
        # (B, embed_dim, H_p, W_p)

        # ── adapter le masque à la grille de patches ────────────────────
        mask = F.interpolate(
            mask_in.unsqueeze(1).float(),
            size=(H_p, W_p),
            mode="nearest",
        ).squeeze(1).bool()                 # (B, H_p, W_p)

        # ── pos nul : déjà intégré dans les tokens ──────────────────────
        pos = torch.zeros_like(src)         # (B, embed_dim, H_p, W_p)

        feature = NestedTensor(src, mask)
        return [feature], [pos]             # listes → features[-1], pos[-1]


def build_vit_backbone(args) -> ViTPatchEmbedder:
    """Factory function appelée depuis sptsv2.py."""
    return ViTPatchEmbedder(
        img_size   = getattr(args, "img_size",   640),
        patch_size = getattr(args, "patch_size",  16),
        in_chans   = 3,
        embed_dim  = getattr(args, "embed_dim",  256),
    )