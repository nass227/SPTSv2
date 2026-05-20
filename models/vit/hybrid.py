import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as tv_models

from util.misc_sptsv2 import NestedTensor


class HybridBackbone(nn.Module):
    """
    Backbone hybride ResNet18 + ViT-Small.
    Charge les poids depuis deux checkpoints SPTSv2 déjà fine-tunés.

    Flux :
      image ──► CNN (ResNet18 layers 0-7)  ──► pool 14×14 ──► proj 512→embed_dim ─┐
                                                                                    ├─ add ──► NestedTensor
      image ──► ViT-Small forward_features ──► tokens 14×14 ──► proj 384→embed_dim ─┘

    Sortie : [NestedTensor(B, embed_dim, 14, 14)], [pos_zeros]
    Compatible avec SPTSv2.forward (attend features[-1].decompose()).
    """

    def __init__(
        self,
        resnet_ckpt: str,
        vit_ckpt: str,
        embed_dim: int = 256,
        freeze_cnn: bool = True,
        freeze_vit: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── ResNet18 ──────────────────────────────────────────────────────
        resnet = tv_models.resnet18(pretrained=False)
        ckpt_r = torch.load(resnet_ckpt, map_location="cpu", weights_only=False)
        state_r = ckpt_r["model"]

        # Dans le checkpoint ResNet, Joiner est nn.Sequential → index 0 est
        # BackboneBase, donc les clés sont backbone.0.body.<nom_layer>.
        cnn_w = {
            k[len("backbone.0.body."):]: v
            for k, v in state_r.items()
            if k.startswith("backbone.0.body.")
        }
        missing, unexpected = resnet.load_state_dict(cnn_w, strict=False)
        print(f"[HybridBackbone] ResNet18 : {len(cnn_w)} clés chargées  "
              f"| missing={len(missing)}  unexpected={len(unexpected)}")

        # Garder seulement le corps convolutif (sans avgpool et fc)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # ── ViT-Small ─────────────────────────────────────────────────────
        self.vit = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=0
        )
        ckpt_v = torch.load(vit_ckpt, map_location="cpu", weights_only=False)
        state_v = ckpt_v["model"]

        # Dans le checkpoint ViT, le modèle timm est sauvegardé sous backbone.vit.*
        vit_w = {
            k[len("backbone.vit."):]: v
            for k, v in state_v.items()
            if k.startswith("backbone.vit.")
        }
        missing_v, unexpected_v = self.vit.load_state_dict(vit_w, strict=False)
        print(f"[HybridBackbone] ViT-Small : {len(vit_w)} clés chargées  "
              f"| missing={len(missing_v)}  unexpected={len(unexpected_v)}")

        vit_dim = self.vit.embed_dim  # 384 pour vit_small

        # ── Couches de fusion (nouvelles, seront entraînées en priorité) ───
        self.proj_cnn = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.proj_vit = nn.Linear(vit_dim, embed_dim)

        # Aligner CNN sur la grille ViT (14×14)
        self.pool = nn.AdaptiveAvgPool2d((14, 14))

        # ── Gel optionnel des branches pré-entraînées ─────────────────────
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad_(False)
            print("[HybridBackbone] CNN gelé.")

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            print("[HybridBackbone] ViT gelé.")

        # Attributs lus par SPTSv2
        self.num_channels = embed_dim
        self.grid_size    = (14, 14)

    def unfreeze_all(self, lr_cnn: float = 1e-6, lr_vit: float = 1e-6):
        """Appeler depuis main pour passer en phase 2."""
        for p in self.cnn.parameters():
            p.requires_grad_(True)
        for p in self.vit.parameters():
            p.requires_grad_(True)
        print(f"[HybridBackbone] Dégel complet. "
              f"Utilise lr_cnn={lr_cnn} lr_vit={lr_vit} dans l'optimiseur.")

    def forward(self, samples: NestedTensor):
        x       = samples.tensors   # (B, 3, H, W)
        mask_in = samples.mask      # (B, H, W)
        B       = x.shape[0]

        # ── Branche CNN ──────────────────────────────────────────────────
        cnn_out  = self.cnn(x)                   # (B, 512, H/32, W/32)
        cnn_out  = self.pool(cnn_out)             # (B, 512, 14, 14)
        cnn_feat = self.proj_cnn(cnn_out)         # (B, embed_dim, 14, 14)

        # ── Branche ViT ──────────────────────────────────────────────────
        x_224   = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        tokens  = self.vit.forward_features(x_224)  # (B, 197, 384)
        tokens  = tokens[:, 1:, :]                   # supprimer CLS → (B, 196, 384)
        tokens  = self.proj_vit(tokens)              # (B, 196, embed_dim)
        vit_feat = tokens.permute(0, 2, 1).reshape(B, self.embed_dim, 14, 14)

        # ── Fusion par addition élémentaire ──────────────────────────────
        fused = cnn_feat + vit_feat                  # (B, embed_dim, 14, 14)

        # ── Masque à la résolution des patches ───────────────────────────
        mask = F.interpolate(
            mask_in.unsqueeze(1).float(), size=(14, 14), mode="nearest"
        ).squeeze(1).bool()                          # (B, 14, 14)

        pos = torch.zeros_like(fused)
        return [NestedTensor(fused, mask)], [pos]
