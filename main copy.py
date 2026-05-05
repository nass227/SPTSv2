# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
python main.py --train_dataset ic13_train --val_dataset ic13_val --data_root "../data" --max_length 25 --batch_size 8 --depths 6 --lr 1e-4 --pre_norm --num_workers 4 --output_dir "./icdar13/output" --train --pad_rec
python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 4 --depths 6 --lr 1e-4 --pre_norm --num_workers 4 --output_dir "./icdar15/output" --train --pad_rec
python main.py --train_dataset ic17_train --val_dataset ic17_val --data_root "../data" --max_length 25 --batch_size 4 --depths 6 --lr 5e-5 --pre_norm --num_workers 4 --output_dir "./icdar17/output" --train --resume "./icdar15/output/checkpoint.pth" --force_lr



python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 8 --depths 6 --lr 1e-4 --lr_backbone 5e-6 --warmup_min_lr 1e-6 --warmup_epochs 10 --min_lr 1e-7 --epochs 150 --pre_norm --num_workers 4 --pad_rec --img_size 512 --patch_size 16 --embed_dim 256 --hidden_dim 256 --nheads 8 --dec_layers 6 --dim_feedforward 1024 --dropout 0.1 --window_size 5 --obj_num 60 --clip_max_norm 0.1 --early_stop --early_stop_patience 30 --output_dir "./icdar15new/output" --device cuda --train
python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 8 --depths 6 --lr 1e-4 --lr_backbone 5e-6 --warmup_min_lr 1e-6 --warmup_epochs 10 --min_lr 1e-7 --epochs 150 --pre_norm --num_workers 4 --pad_rec --img_size 512 --patch_size 16 --embed_dim 256 --hidden_dim 256 --nheads 8 --dec_layers 6 --dim_feedforward 1024 --dropout 0.1 --window_size 5 --obj_num 60 --clip_max_norm 0.1 --early_stop --early_stop_patience 30 --output_dir "./icdar15new/output" --device cuda --train
python main.py --train_dataset ic17_train --val_dataset ic17_val --data_root "../data" --resume "./icdar17new/output/checkpoint.pth" --force_lr --lr 5e-5 --lr_backbone 1e-6 --warmup_min_lr 1e-6 --warmup_epochs 5 --min_lr 1e-7 --epochs 400 --batch_size 8 --depths 6 --pre_norm --num_workers 4 --pad_rec --img_size 512 --patch_size 16 --embed_dim 256 --hidden_dim 256 --nheads 8 --dec_layers 6 --dim_feedforward 1024 --dropout 0.1 --window_size 5 --obj_num 60 --clip_max_norm 0.1 --early_stop --early_stop_patience 30 --output_dir "./icdar17new/output" --device cuda --train


CUDA_VISIBLE_DEVICES=1 python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --pretrained_vit --lr 5e-4 --lr_backbone 1e-5 --warmup_min_lr 1e-6 --warmup_epochs 10 --min_lr 1e-5 --epochs 150 --batch_size 16 --depths 6 --pre_norm --num_workers 4 --pad_rec --img_size 224 --patch_size 16 --embed_dim 256 --hidden_dim 256 --nheads 8 --dec_layers 6 --dim_feedforward 1024 --dropout 0.1 --window_size 5 --obj_num 60 --early_stop --early_stop_patience 30 --output_dir "./ic15_pretrained/output" --device cuda --train 
CUDA_VISIBLE_DEVICES=1 python main.py --train_dataset ic17_train --val_dataset ic17_val --data_root "../data" --resume "./ic15_pretrained/output/checkpoint.pth" --force_lr --finetune --lr 1e-5 --lr_backbone 1e-6 --warmup_epochs 0 --min_lr 1e-6 --epochs 400 --batch_size 16 --depths 6 --pre_norm --num_workers 4 --pad_rec --img_size 224 --patch_size 16 --embed_dim 256 --hidden_dim 256 --nheads 8 --dec_layers 6 --dim_feedforward 1024 --dropout 0.1 --window_size 5 --obj_num 60 --early_stop --early_stop_patience 30 --output_dir "./ic17_pretrained/output" --device cuda --train
"""


import time
import json
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc_sptsv2 as utils
from util.data import process_args
from datasets import build_dataset
from engine_sptsv2 import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set SPTSv2-ViT', add_help=False)
    parser.add_argument('--pretrained_vit', action='store_true',
                    help='Utiliser ViT pré-entraîné sur ImageNet')
    # ── Optimisation ────────────────────────────────────────────────────
    parser.add_argument('--lr',            default=1e-4, type=float)
    parser.add_argument('--lr_backbone',   default=5e-6, type=float)
    parser.add_argument('--batch_size',    default=2,    type=int)
    parser.add_argument('--weight_decay',  default=1e-4, type=float)
    parser.add_argument('--epochs',        default=300,  type=int)
    parser.add_argument('--clip_max_norm', default=0.1,  type=float)

    # ── Warmup & LR schedule ────────────────────────────────────────────
    parser.add_argument('--warmup_min_lr', default=1e-6, type=float)
    parser.add_argument('--min_lr',        default=1e-7, type=float)
    parser.add_argument('--warmup_epochs', default=10,   type=int)

    # ── Early stopping ──────────────────────────────────────────────────
    parser.add_argument('--early_stop',          action='store_true',
                        help='Active l early stopping')
    parser.add_argument('--early_stop_patience', default=20,  type=int,
                        help='Nombre d epochs sans amélioration avant arrêt')
    parser.add_argument('--early_stop_delta',    default=1e-4, type=float,
                        help='Amélioration minimale considérée significative')

    # ── Modes ───────────────────────────────────────────────────────────
    parser.add_argument('--train',     action='store_true')
    parser.add_argument('--eval',      action='store_true')
    parser.add_argument('--finetune',  action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--force_lr',  action='store_true',
                        help='Forcer le nouveau LR après un resume')

    # ── Chemins ─────────────────────────────────────────────────────────
    parser.add_argument('--code_dir',       type=str, default='.')
    parser.add_argument('--output_dir',     default='')
    parser.add_argument('--resume',         default='')
    parser.add_argument('--frozen_weights', type=str, default=None)

    # ── Logs ────────────────────────────────────────────────────────────
    parser.add_argument('--print_freq', type=int, default=10)

    # ── Données ─────────────────────────────────────────────────────────

    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument('--padding_bins', type=int, default=0)
    parser.add_argument('--num_box', type=int, default=60)
    parser.add_argument('--pts_key', type=str, default='center_pts')
    parser.add_argument('--no_known_char', type=int, default=95)
    parser.add_argument('--pad_rec_index', type=int, default=96)
    parser.add_argument('--pad_rec', action='store_true')
    parser.add_argument('--dict_name', type=str, default='en_US.dic')
    parser.add_argument('--use_dict', action='store_true')

    # ── Augmentations ───────────────────────────────────────────────────
    parser.add_argument('--max_size_train',   type=int,   default=1600)
    parser.add_argument('--min_size_train',   type=int,   nargs='+',
                        default=[512, 640, 672, 704, 736, 768, 800, 832, 864, 896])
    parser.add_argument('--max_size_test',    type=int,   default=1824)
    parser.add_argument('--min_size_test',    type=int,   default=1024)
    parser.add_argument('--crop_min_ratio',   type=float, default=0.5)
    parser.add_argument('--crop_max_ratio',   type=float, default=1.0)
    parser.add_argument('--crop_prob',        type=float, default=1.0)
    parser.add_argument('--rotate_max_angle', type=int,   default=30)
    parser.add_argument('--rotate_prob',      type=float, default=0.3)
    parser.add_argument('--brightness',       type=float, default=0.5)
    parser.add_argument('--contrast',         type=float, default=0.5)
    parser.add_argument('--saturation',       type=float, default=0.5)
    parser.add_argument('--hue',              type=float, default=0.5)
    parser.add_argument('--distortion_prob',  type=float, default=0.5)

    # ── Backbone ViT ────────────────────────────────────────────────────
    parser.add_argument('--img_size',           type=int, default=512)
    parser.add_argument('--patch_size',         type=int, default=16)
    parser.add_argument('--embed_dim',          type=int, default=256)
    parser.add_argument('--dilation',           action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'))

    # ── Transformer ─────────────────────────────────────────────────────
    parser.add_argument('--enc_layers',       default=6,    type=int)
    parser.add_argument('--dec_layers',       default=6,    type=int)
    parser.add_argument('--window_size',      default=5,    type=int)
    parser.add_argument('--obj_num',          default=60,   type=int)
    parser.add_argument('--max_length',       default=25,   type=int)
    parser.add_argument('--dim_feedforward',  default=1024, type=int)
    parser.add_argument('--hidden_dim',       default=256,  type=int)
    parser.add_argument('--dropout',          default=0.1,  type=float)
    parser.add_argument('--depths',           default=6,    type=int)
    parser.add_argument('--nheads',           default=8,    type=int)
    parser.add_argument('--num_queries',      default=100,  type=int)
    parser.add_argument('--pre_norm',         action='store_true')
    parser.add_argument('--transformer_type', type=str, default='vanilla')

    # ── Dataset ─────────────────────────────────────────────────────────
    parser.add_argument('--dataset_file',     default='ocr')
    parser.add_argument('--train_dataset',    type=str)
    parser.add_argument('--val_dataset',      type=str)
    parser.add_argument('--data_root',        type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # ── Divers ──────────────────────────────────────────────────────────
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=42,  type=int)
    parser.add_argument('--start_epoch', default=0,   type=int, metavar='N')
    parser.add_argument('--num_workers', default=2,   type=int)
    parser.add_argument('--masks',       action='store_true')

    # ── Distribué ───────────────────────────────────────────────────────
    parser.add_argument('--world_size',  default=1,       type=int)
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--local_rank',  default=0,       type=int)

    return parser


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Arrête l'entraînement si la loss ne s'améliore pas
    pendant `patience` epochs consécutives.
    """
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
            # amélioration suffisante → réinitialiser le compteur
            self.best_loss = loss
            self.counter   = 0
        else:
            # pas d'amélioration
            self.counter += 1
            print(f"EarlyStopping : {self.counter}/{self.patience} "
                  f"(best={self.best_loss:.4f}, current={loss:.4f})")
            if self.counter >= self.patience:
                self.stop = True
                print("Early stopping déclenché !")

    @property
    def improved(self):
        return self.counter == 0


# ─────────────────────────────────────────────
# LR Schedule
# ─────────────────────────────────────────────
def build_lr_schedule(args):
    """
    Warmup linéaire + decay linéaire (comme SPTSv2 original).
    LR reste élevé plus longtemps → meilleure convergence.
    """
    if args.finetune:
        return [args.lr] * args.epochs

    # ── Warmup linéaire ─────────────────────────────────────────────────
    warmup_lr = [
        args.warmup_min_lr
        + (args.lr - args.warmup_min_lr) * i / max(args.warmup_epochs, 1)
        for i in range(args.warmup_epochs)
    ]

    # ── Decay linéaire : lr → min_lr ────────────────────────────────────
    decay_epochs = args.epochs - args.warmup_epochs
    linear_lr = [
        args.lr - (args.lr - args.min_lr) * i / max(decay_epochs, 1)
        for i in range(decay_epochs)
    ]

    return warmup_lr + linear_lr

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────
def save_checkpoint(output_dir, model_without_ddp, optimizer, epoch, args,
                    save_full=False, is_best=False):
    """
    Sauvegarde :
    - checkpoint.pth         : dernier checkpoint (écrasé à chaque epoch)
    - checkpoint{epoch}.pth  : tous les 10 epochs seulement
    - model_epoch{epoch}.pt  : modèle seul tous les 10 epochs
    - best_model.pt          : meilleur modèle selon l'early stopping
    """
    checkpoint = {
        'model':     model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     epoch,
        'args':      args,
    }

    # ── toujours écrasé ─────────────────────────────────────────────────
    utils.save_on_master(checkpoint, output_dir / 'checkpoint.pth')

    # ── tous les 10 epochs ──────────────────────────────────────────────
    if save_full:
        utils.save_on_master(
            checkpoint,
            output_dir / f'checkpoint{epoch:04d}.pth'
        )
        utils.save_on_master(
            model_without_ddp.state_dict(),
            output_dir / f'model_epoch{epoch:04d}.pt'
        )
        print(f"Checkpoint et modèle sauvegardés : epoch {epoch:04d}")

    # ── meilleur modèle ─────────────────────────────────────────────────
    if is_best:
        utils.save_on_master(
            model_without_ddp.state_dict(),
            output_dir / 'best_model.pt'
        )
        print(f"Meilleur modèle sauvegardé : best_model.pt (epoch {epoch})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    args = process_args(args)
    print(args)

    device = torch.device(args.device)

    # ── Reproductibilité ────────────────────────────────────────────────
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Modèle ──────────────────────────────────────────────────────────
    model, criterion = build_model(args)
    model.to(device)
    print(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Nombre de paramètres entraînables : {n_parameters:,}')

    # ── Optimiseur ──────────────────────────────────────────────────────
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Datasets ────────────────────────────────────────────────────────
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val   = build_dataset(image_set='val',   args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False) \
                        if dataset_val is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val) \
                        if dataset_val is not None else None

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler = batch_sampler_train,
        collate_fn    = utils.collate_fn(args),
        num_workers   = args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size  = args.batch_size,
        sampler     = sampler_val,
        drop_last   = False,
        collate_fn  = utils.collate_fn(args),
        num_workers = args.num_workers,
    ) if dataset_val is not None else None

    # ── Poids figés ─────────────────────────────────────────────────────
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # ── Resume ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(
                args.resume, map_location='cpu', weights_only=False
            )
        model_without_ddp.load_state_dict(checkpoint['model'])

        if (not args.eval
                and 'optimizer' in checkpoint
                and 'epoch'     in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Reprise depuis l'epoch {args.start_epoch}")

        if args.force_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.lr_backbone
            print(f"LR forcé → transformer: {args.lr}  backbone: {args.lr_backbone}")

    # ── Évaluation seule ────────────────────────────────────────────────
    if args.eval:
        if dataset_val is None:
            print('Dataset de validation introuvable !')
            return
        evaluate(
            model, criterion,
            data_loader_val, device,
            args.output_dir, args.chars,
            args.start_index, args.visualize,
            args.max_length,
        )
        return

    # ── Entraînement ────────────────────────────────────────────────────
    print("Début de l'entraînement")
    start_time = time.time()

    if not args.finetune:
        learning_rate_schedule = build_lr_schedule(args)
    else:
        learning_rate_schedule = [args.lr] * args.epochs

    # ── Early stopping ──────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience = args.early_stop_patience,
        delta    = args.early_stop_delta,
    ) if args.early_stop else None

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion,
            data_loader_train, optimizer,
            device, epoch,
            args.clip_max_norm,
            learning_rate_schedule,
            args.print_freq,
            args.max_length,
        )

        current_loss = train_stats['loss']

        # ── Early stopping check ─────────────────────────────────────────
        if early_stopping is not None:
            early_stopping.step(current_loss)
            is_best = early_stopping.improved
        else:
            is_best = False

        # ── Sauvegarde ──────────────────────────────────────────────────
        if args.output_dir:
            save_full = ((epoch + 1) % 10 == 0)
            save_checkpoint(
                output_dir, model_without_ddp, optimizer,
                epoch, args,
                save_full = save_full,
                is_best   = is_best,
            )

        # ── Log ─────────────────────────────────────────────────────────
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch':        epoch,
            'n_parameters': n_parameters,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

        # ── Arrêt anticipé ──────────────────────────────────────────────
        if early_stopping is not None and early_stopping.stop:
            print(f"Entraînement arrêté à l'epoch {epoch} par early stopping.")
            break

    total_time     = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Temps total d'entraînement : {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'SPTSv2-ViT', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)