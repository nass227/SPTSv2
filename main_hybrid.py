"""
Entraînement SPTSv2 avec backbone hybride ResNet18 + ViT-Small.
Les deux backbones sont initialisés depuis tes propres checkpoints fine-tunés.

Exemple d'utilisation :

  # Phase 1 : backbone gelé, seulement proj + decoder (rapide, ~20 epochs)
  python main_hybrid.py \
      --train_dataset icdarall_train --data_root ./icall \
      --resnet_ckpt ./output/resnet/best_model.pt \
      --vit_ckpt    ./output/vit/best_model.pt \
      --freeze_backbone_epochs 20 \
      --lr 1e-4 --lr_backbone 1e-5 \
      --warmup_epochs 5 --min_lr 1e-6 \
      --epochs 80 --batch_size 16 \
      --embed_dim 256 --hidden_dim 256 \
      --nheads 8 --dec_layers 6 --dim_feedforward 1024 \
      --pad_rec --pre_norm --early_stop --early_stop_patience 20 \
      --val_split 0.2 --num_workers 8 \
      --output_dir ./output/hybrid --device cuda --train

  # Reprendre depuis un checkpoint hybrid
  python main_hybrid.py ... --resume ./output/hybrid/checkpoint.pth
"""

import time
import json
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Subset

import datasets
import util.misc_sptsv2 as utils
from util.data import process_args
from datasets import build_dataset
from engine_sptsv2 import evaluate, train_one_epoch, validate_loss

# Imports modèles
from models.vit.hybrid import HybridBackbone
from models.vit.sptsv2 import SPTSv2
from models.encoder_decoder import build_transformer


# ─────────────────────────────────────────────────────────────────────────────
# Construction du modèle hybride
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_model(args):
    device = torch.device(args.device)

    freeze = not args.no_freeze_start

    backbone = HybridBackbone(
        resnet_ckpt = args.resnet_ckpt,
        vit_ckpt    = args.vit_ckpt,
        embed_dim   = args.embed_dim,
        freeze_cnn  = freeze,
        freeze_vit  = freeze,
    )

    transformer = build_transformer(args)

    num_classes = args.padding_index + 1
    model = SPTSv2(backbone, transformer, num_classes)

    weight = torch.ones(num_classes)
    weight[args.end_index]   = 0.01
    weight[args.noise_index] = 0.01
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
    criterion.to(device)

    return model, criterion


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("SPTSv2-Hybrid", add_help=False)

    # ── Chemins backbones ────────────────────────────────────────────────
    parser.add_argument("--resnet_ckpt", type=str, required=True,
                        help="Checkpoint SPTSv2 entraîné avec ResNet18 "
                             "(contient la clé 'model')")
    parser.add_argument("--vit_ckpt", type=str, required=True,
                        help="Checkpoint SPTSv2 entraîné avec ViT-Small "
                             "(contient la clé 'model')")

    # ── Stratégie de dégel ───────────────────────────────────────────────
    parser.add_argument("--freeze_backbone_epochs", type=int, default=20,
                        help="Nombre d'epochs où CNN+ViT restent gelés "
                             "(phase 1). Après, tout est dégelé avec lr_backbone.")
    parser.add_argument("--no_freeze_start", action="store_true",
                        help="Ne pas geler le backbone au départ "
                             "(tout entraîner depuis le début)")

    # ── Mixed precision ──────────────────────────────────────────────────
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (FP16) training")

    # ── Optimisation ────────────────────────────────────────────────────
    parser.add_argument("--lr",            default=1e-4,  type=float)
    parser.add_argument("--lr_backbone",   default=1e-5,  type=float)
    parser.add_argument("--batch_size",    default=8,     type=int)
    parser.add_argument("--weight_decay",  default=1e-4,  type=float)
    parser.add_argument("--epochs",        default=80,    type=int)
    parser.add_argument("--clip_max_norm", default=0.1,   type=float)

    # ── Warmup & LR schedule (cosine) ───────────────────────────────────
    parser.add_argument("--warmup_min_lr", default=1e-6, type=float)
    parser.add_argument("--min_lr",        default=1e-7, type=float)
    parser.add_argument("--warmup_epochs", default=10,   type=int)

    # ── Early stopping ───────────────────────────────────────────────────
    parser.add_argument("--early_stop",          action="store_true")
    parser.add_argument("--early_stop_patience", default=20,   type=int)
    parser.add_argument("--early_stop_delta",    default=1e-4, type=float)

    # ── Validation split ─────────────────────────────────────────────────
    parser.add_argument("--val_split", default=0.2, type=float)

    # ── Modes ────────────────────────────────────────────────────────────
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--finetune",  action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force_lr",  action="store_true")

    # ── Chemins ──────────────────────────────────────────────────────────
    parser.add_argument("--code_dir",       type=str, default=".")
    parser.add_argument("--output_dir",     default="")
    parser.add_argument("--resume",         default="")
    parser.add_argument("--frozen_weights", type=str, default=None)

    # ── Logs ─────────────────────────────────────────────────────────────
    parser.add_argument("--print_freq", type=int, default=10)

    # ── Données ──────────────────────────────────────────────────────────
    parser.add_argument("--bins",          type=int, default=1000)
    parser.add_argument("--chars",         type=str,
                        default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                '[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument("--padding_bins",  type=int,   default=0)
    parser.add_argument("--num_box",       type=int,   default=60)
    parser.add_argument("--pts_key",       type=str,   default="center_pts")
    parser.add_argument("--no_known_char", type=int,   default=95)
    parser.add_argument("--pad_rec_index", type=int,   default=96)
    parser.add_argument("--pad_rec",       action="store_true")
    parser.add_argument("--dict_name",     type=str,   default="en_US.dic")
    parser.add_argument("--use_dict",      action="store_true")

    # ── Augmentations ─────────────────────────────────────────────────────
    parser.add_argument("--max_size_train",   type=int,   default=1600)
    parser.add_argument("--min_size_train",   type=int,   nargs="+",
                        default=[512, 640, 672, 704, 736, 768, 800])
    parser.add_argument("--max_size_test",    type=int,   default=1824)
    parser.add_argument("--min_size_test",    type=int,   default=1024)
    parser.add_argument("--crop_min_ratio",   type=float, default=0.5)
    parser.add_argument("--crop_max_ratio",   type=float, default=1.0)
    parser.add_argument("--crop_prob",        type=float, default=1.0)
    parser.add_argument("--rotate_max_angle", type=int,   default=30)
    parser.add_argument("--rotate_prob",      type=float, default=0.3)
    parser.add_argument("--brightness",       type=float, default=0.5)
    parser.add_argument("--contrast",         type=float, default=0.5)
    parser.add_argument("--saturation",       type=float, default=0.5)
    parser.add_argument("--hue",              type=float, default=0.5)
    parser.add_argument("--distortion_prob",  type=float, default=0.5)

    # ── Backbone / ViT ────────────────────────────────────────────────────
    parser.add_argument("--img_size",           type=int, default=224)
    parser.add_argument("--patch_size",         type=int, default=16)
    parser.add_argument("--embed_dim",          type=int, default=256)
    parser.add_argument("--dilation",           action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str,
                        choices=("sine", "learned"))

    # ── Transformer ───────────────────────────────────────────────────────
    parser.add_argument("--enc_layers",       default=6,    type=int)
    parser.add_argument("--dec_layers",       default=6,    type=int)
    parser.add_argument("--window_size",      default=5,    type=int)
    parser.add_argument("--obj_num",          default=60,   type=int)
    parser.add_argument("--max_length",       default=25,   type=int)
    parser.add_argument("--dim_feedforward",  default=1024, type=int)
    parser.add_argument("--hidden_dim",       default=256,  type=int)
    parser.add_argument("--dropout",          default=0.1,  type=float)
    parser.add_argument("--depths",           default=6,    type=int)
    parser.add_argument("--nheads",           default=8,    type=int)
    parser.add_argument("--num_queries",      default=100,  type=int)
    parser.add_argument("--pre_norm",         action="store_true")
    parser.add_argument("--transformer_type", type=str, default="vanilla")

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument("--dataset_file",     default="ocr")
    parser.add_argument("--train_dataset",    type=str)
    parser.add_argument("--val_dataset",      type=str)
    parser.add_argument("--data_root",        type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    # ── Divers ────────────────────────────────────────────────────────────
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--start_epoch", default=0,   type=int, metavar="N")
    parser.add_argument("--num_workers", default=2,   type=int)
    parser.add_argument("--masks",       action="store_true")

    # ── Distribué ─────────────────────────────────────────────────────────
    parser.add_argument("--world_size",  default=1,       type=int)
    parser.add_argument("--dist_url",    default="env://")
    parser.add_argument("--local_rank",  default=0,       type=int)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (cosine, comme main_vit.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_schedule(args):
    if args.finetune:
        return [args.lr] * args.epochs
    warmup_lr = [
        args.warmup_min_lr
        + (args.lr - args.warmup_min_lr) * i / max(args.warmup_epochs, 1)
        for i in range(args.warmup_epochs)
    ]
    cosine_epochs = args.epochs - args.warmup_epochs
    cosine_lr = [
        args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1 + np.cos(np.pi * i / max(cosine_epochs, 1))
        )
        for i in range(cosine_epochs)
    ]
    return warmup_lr + cosine_lr


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=20, delta=1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best_loss = None
        self.counter   = 0
        self.stop      = False

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"EarlyStopping : {self.counter}/{self.patience} "
                  f"(best={self.best_loss:.4f}, current={loss:.4f})")
            if self.counter >= self.patience:
                self.stop = True
                print("Early stopping déclenché !")

    @property
    def improved(self):
        return self.counter == 0


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(output_dir, model_without_ddp, optimizer, epoch, args,
                    save_full=False, is_best=False, scaler=None):
    checkpoint = {
        "model":     model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch":     epoch,
        "args":      args,
        "scaler":    scaler.state_dict() if scaler is not None else None,
    }
    utils.save_on_master(checkpoint, output_dir / "checkpoint.pth")
    if save_full:
        utils.save_on_master(checkpoint, output_dir / f"checkpoint{epoch:04d}.pth")
        utils.save_on_master(model_without_ddp.state_dict(),
                             output_dir / f"model_epoch{epoch:04d}.pt")
        print(f"Checkpoint sauvegardé : epoch {epoch:04d}")
    if is_best:
        utils.save_on_master(model_without_ddp.state_dict(),
                             output_dir / "best_model.pt")
        print(f"Meilleur modèle : best_model.pt (epoch {epoch})")


# ─────────────────────────────────────────────────────────────────────────────
# Optimiseur : construit selon la phase (backbone gelé ou non)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model_without_ddp, args):
    """
    Exactement 2 groupes de paramètres (requis par train_one_epoch dans engine_sptsv2.py
    qui hardcode param_groups[0] et param_groups[1]) :
      - groupe 0 : non-backbone (decoder, heads, proj_cnn, proj_vit)  → lr principal
      - groupe 1 : backbone CNN + ViT bruts                           → lr_backbone
        (vide en phase 1 car gelés, AdamW gère les groupes vides)
    """
    non_backbone_params = [
        p for n, p in model_without_ddp.named_parameters()
        if "backbone.cnn" not in n and "backbone.vit" not in n and p.requires_grad
    ]
    backbone_params = [
        p for n, p in model_without_ddp.named_parameters()
        if ("backbone.cnn" in n or "backbone.vit" in n) and p.requires_grad
    ]

    param_dicts = [
        {"params": non_backbone_params},
        {"params": backbone_params, "lr": args.lr_backbone},
    ]
    return torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

import torch.nn as nn  # noqa: E402 (nn utilisé dans build_hybrid_model)


def main(args):
    utils.init_distributed_mode(args)
    args = process_args(args)
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_hybrid_model(args)
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres total : {n_total:,}  |  entraînables : {n_train:,}")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if scaler is not None:
        print("Mixed precision (AMP) activé.")

    # ── Datasets ─────────────────────────────────────────────────────────
    use_val_split = (not args.val_dataset) and (args.val_split > 0)
    if use_val_split:
        full_ds = build_dataset(image_set="train", args=args)
        n_val   = max(1, int(len(full_ds) * args.val_split))
        n_tr    = len(full_ds) - n_val
        g = torch.Generator().manual_seed(args.seed)
        dataset_train, dataset_val_loss = torch.utils.data.random_split(
            full_ds, [n_tr, n_val], generator=g
        )
        print(f"Val split : {n_tr} train | {n_val} val")
    else:
        dataset_train    = build_dataset(image_set="train", args=args)
        dataset_val_loss = None

    dataset_val = (
        build_dataset(image_set="val", args=args) if args.val_dataset else None
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn(args),
        num_workers=args.num_workers,
    )

    data_loader_val_loss = DataLoader(
        dataset_val_loss,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset_val_loss),
        drop_last=False,
        collate_fn=utils.collate_fn(args),
        num_workers=args.num_workers,
    ) if dataset_val_loss is not None else None

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        drop_last=False,
        collate_fn=utils.collate_fn(args),
        num_workers=args.num_workers,
    ) if dataset_val is not None else None

    # ── Resume ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(ckpt["model"])
        if not args.eval and "optimizer" in ckpt and "epoch" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            args.start_epoch = ckpt["epoch"] + 1
            print(f"Reprise depuis epoch {args.start_epoch}")
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if args.force_lr:
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            print(f"LR forcé → {args.lr}")

    # ── Eval seule ───────────────────────────────────────────────────────
    if args.eval:
        if dataset_val is None:
            print("Dataset de validation introuvable !")
            return
        evaluate(model, criterion, data_loader_val, device,
                 args.output_dir, args.chars, args.start_index,
                 args.category_start_index, args.visualize, args.max_length)
        return

    print("Début de l'entraînement")
    print(f"  Phase 1 (backbone gelé) : epochs 0 → {args.freeze_backbone_epochs - 1}")
    print(f"  Phase 2 (tout dégelé)   : epochs {args.freeze_backbone_epochs} → {args.epochs - 1}")

    start_time = time.time()
    learning_rate_schedule = build_lr_schedule(args)

    early_stopping = EarlyStopping(
        patience=args.early_stop_patience,
        delta=args.early_stop_delta,
    ) if args.early_stop else None

    phase2_started = args.start_epoch >= args.freeze_backbone_epochs

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # ── Passage en phase 2 ───────────────────────────────────────────
        if (not args.no_freeze_start
                and not phase2_started
                and epoch >= args.freeze_backbone_epochs):
            print(f"\n>>> Epoch {epoch} : passage en phase 2 — dégel backbone <<<")
            model_without_ddp.backbone.unfreeze_all(
                lr_cnn=args.lr_backbone,
                lr_vit=args.lr_backbone,
            )
            # Reconstruire l'optimiseur pour inclure les nouveaux params dégelés
            optimizer = build_optimizer(model_without_ddp, args)
            # AMP scaler survive le rebuild de l'optimiseur sans action
            phase2_started = True

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, args.clip_max_norm,
            learning_rate_schedule, args.print_freq, args.max_length,
            scaler=scaler,
        )

        # ── Validation loss ──────────────────────────────────────────────
        val_stats = None
        if data_loader_val_loss is not None:
            val_stats = validate_loss(
                model, criterion, data_loader_val_loss,
                device, epoch, args.max_length,
            )

        current_loss = (
            val_stats["loss"] if val_stats is not None
            else train_stats["loss"]
        )

        if early_stopping is not None:
            early_stopping.step(current_loss)
            is_best = early_stopping.improved
        else:
            is_best = False

        if args.output_dir:
            save_checkpoint(
                output_dir, model_without_ddp, optimizer, epoch, args,
                save_full=((epoch + 1) % 10 == 0),
                is_best=is_best,
                scaler=scaler,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **(  {f"val_{k}": v for k, v in val_stats.items()}
                 if val_stats is not None else {}),
            "epoch":        epoch,
            "phase":        2 if phase2_started else 1,
            "n_parameters": n_train,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        val_loss_str = f"{val_stats['loss']:.4f}" if val_stats else "n/a"
        print(f"Epoch {epoch:3d} [ph{'2' if phase2_started else '1'}]  "
              f"train={train_stats['loss']:.4f}  val={val_loss_str}  "
              f"{'★ BEST' if is_best else ''}")

        if early_stopping is not None and early_stopping.stop:
            print(f"Early stopping à l'epoch {epoch}.")
            break

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Temps total : {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPTSv2-Hybrid", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
