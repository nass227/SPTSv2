# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.misc_sptsv2 import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)
from .encoder_decoder import build_transformer
from .vit import build_vit_backbone          # ← import du nouveau backbone


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SPTSv2(nn.Module):
    """
    Parameters
    ----------
    backbone    : ViTPatchEmbedder  (expose .num_channels)
    transformer : encoder-decoder SPTSv2
    num_classes : taille du vocabulaire (padding_index + 1)
    """

    def __init__(self, backbone: nn.Module, transformer: nn.Module, num_classes: int):
        super().__init__()
        self.backbone    = backbone
        self.transformer = transformer
        self.num_classes = num_classes

        hidden_dim = transformer.d_model
        self.vocab_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        # 1×1 conv : embed_dim → hidden_dim
        self.input_proj  = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(
        self,
        samples,
        sequence,
        sequence_reg,
        text_length: int = 25,
    ):
        """
        Parameters
        ----------
        samples      : NestedTensor | list[Tensor] | Tensor
        sequence     : séquence de localisation   (1er décodeur)
        sequence_reg : séquence de reconnaissance (2e  décodeur)
        text_length  : longueur max d'un mot (défaut 25)

        Returns (training)
        ------------------
        out_point : logits de position
        out_label : logits de caractères  (B, nb_mots, num_classes)

        Returns (inférence)
        -------------------
        out       : points + labels concaténés
        out_v     : valeurs brutes (points + labels)
        rec_score : scores de reconnaissance  (B, nb_mots, vocab)
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None, "Le masque ne doit pas être None."

        hs = self.transformer(
            self.input_proj(src), mask, None, pos[-1],
            sequence, sequence_reg, self.vocab_embed,
        )

        if hs is None:
            return None

        B = src.size(0)

        if self.training:
            assert hs[2].size(2) == (text_length + 2), (
                f"Dimension inattendue : {hs[2].size(2)} ≠ {text_length + 2}"
            )
            out_point = self.vocab_embed(hs[0])[-1]
            out_label = (
                self.vocab_embed(hs[2])[-1][:, 1 : text_length + 1]
                .reshape(B, -1, self.num_classes)
            )
            return out_point, out_label

        else:
            out_point = hs[0].reshape(B, -1, 2)
            out_label = hs[2].reshape(B, -1, text_length)
            out       = torch.cat([out_point, out_label], dim=-1).view(B, -1)

            value     = hs[3].reshape(B, -1, 2)
            value_reg = hs[1].reshape(B, -1, text_length)
            out_v     = torch.cat([value, value_reg], dim=-1).view(B, -1)

            rec_score = hs[4].reshape(B, -1, hs[4].shape[-1])
            return out, out_v, rec_score


def build(args):
    device = torch.device(args.device)

    backbone    = build_vit_backbone(args)    # ← depuis vit.py
    transformer = build_transformer(args)

    num_classes = args.padding_index + 1
    model = SPTSv2(backbone, transformer, num_classes)

    weight = torch.ones(num_classes)
    weight[args.end_index]   = 0.01
    weight[args.noise_index] = 0.01
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
    criterion.to(device)

    return model, criterion