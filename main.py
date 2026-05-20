# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 8 --lr 1e-4 --lr_backbone 1e-5 --warmup_epochs 10 --min_lr 1e-7 --epochs 200 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results/icdar15_new" --device cuda --train --amp

python main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 8 --lr 1e-4 --lr_backbone 1e-5 --warmup_epochs 10 --min_lr 1e-7 --epochs 200 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results/icdar15_new" --device cuda --train --amp
main.py --train_dataset ic15_train --val_dataset ic15_val --data_root "../data" --max_length 25 --batch_size 8 --lr 8e-5 --lr_backbone 8e-6 --warmup_epochs 10 --min_lr 1e-7 --epochs 200 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results/icdar15_new" --device cuda --train --amp --resume "./results/icdar15_new/checkpoint.pth" 

 python main.py --train_dataset ic17_train --val_dataset ic17_val --data_root "../data" --resume "./results/icdar15_new/checkpoint0149.pth" --finetune --lr 2e-5 --lr_backbone 2e-6 --epochs 100 --warmup_epochs 0 --min_lr 1e-6 --batch_size 8 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results/icdar2017" --train --amp

python main.py --train_dataset ic17_train  --val_dataset ic17_val  --data_root "../data" --resume "./results/icdar2017/checkpoint.pth" --lr 1e-5 --lr_backbone 1e-6  --epochs 100  --warmup_epochs 0  --min_lr 1e-6 --finetune --batch_size 8 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results/icdar2017" --train --amp 

nohup python main.py --train_dataset icdarall_train --val_dataset icdarall_test --data_root "../new_data" --batch_size 8 --lr 5e-4 --lr_backbone 1e-5 --warmup_epochs 10 --warmup_min_lr 1e-7 --min_lr 1e-5 --epochs 250 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results_new/icdarall" --device cuda --train > train.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python main.py --train_dataset icdarall_train --val_dataset icdarall_test --data_root "../new_data" --lr 5e-4 --lr_backbone 1e-5 --epochs 250 --warmup_epochs 10 --warmup_min_lr 1e-7 --min_lr 1e-5 --batch_size 8 --pre_norm --num_workers 4 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir "./results_new/icdarall" --train --amp > train.log 2>&1 &

on colab 
!python3.10 main.py --train_dataset ICDAR2019_train --val_dataset ICDAR2019_test --data_root  "/content/Data" --lr 5e-4 --lr_backbone 1e-5 --epochs 250 --warmup_epochs 10 --warmup_min_lr 1e-7 --min_lr 1e-5 --batch_size 8 --pre_norm --num_workers 1 --pad_rec --early_stop --early_stop_patience 30 --early_stop_delta 1e-4 --output_dir  "/content/results_new" --train --resume "/content/checkpoint.pth" --amp --max_size_train 640 --min_size_train 320 384 448 512 --max_size_test 768 --min_size_test 512

 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python SPTSv2/main.py  --train  --train_dataset icdarall_train --data_root /workspace/pfefn/icall  --output_dir ./output/resnet --epochs 150 --batch_size 20 --lr 5e-5 --lr_backbone 1e-6 --pad_rec --weight_decay 1e-4 --warmup_epochs 5 --dropout 0.1 --early_stop --early_stop_patience 30 --val_split 0.2  --num_workers 8 --amp > train.log 2>&1 &
 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python SPTSv2/main.py  --train  --train_dataset icdarall_train --data_root /workspace/pfefn/icall  --output_dir ./output/resnet --epochs 150 --batch_size 20 --lr 5e-5 --lr_backbone 1e-6 --pad_rec --weight_decay 1e-4 --warmup_epochs 5 --dropout 0.1 --early_stop --early_stop_patience 30 --val_split 0.2  --num_workers 8 --amp --resume "/workspace/pfefn/output/resnet/checkpoint.pth"> train2.log 2>&1 &
 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup python SPTSv2/main.py  --train  --train_dataset icdarall_train --data_root /workspace/pfefn/icall  --output_dir ./output/resnet --epochs 150 --batch_size 20 --lr 5e-5 --lr_backbone 1e-6 --lr_drop 120 --pad_rec --weight_decay 1e-4 --warmup_epochs 5 --dropout 0.1 --early_stop --early_stop_patience 30 --val_split 0.2  --num_workers 8 --amp --resume "/workspace/pfefn/output/resnet/checkpoint.pth"> train3.log 2>&1 &

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
from engine_sptsv2 import evaluate, train_one_epoch, validate_loss
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('SPTSv2-ResNet', add_help=False)

    # ── Optimisation ────────────────────────────────────────────────────
    parser.add_argument('--lr',            default=1e-4,  type=float)
    parser.add_argument('--lr_backbone',   default=1e-5,  type=float)
    parser.add_argument('--batch_size',    default=4,     type=int)
    parser.add_argument('--weight_decay',  default=1e-4,  type=float)
    parser.add_argument('--epochs',        default=250,   type=int)
    parser.add_argument('--lr_drop',       default=200,   type=int)
    parser.add_argument('--clip_max_norm', default=0.1,   type=float,
                        help='gradient clipping max norm')

    # ── Warmup & LR schedule (linear warmup + linear decay; same as main_original) ──
    parser.add_argument('--warmup_min_lr', default=0.0001,  type=float)
    parser.add_argument('--min_lr',        default=0.00001, type=float)
    parser.add_argument('--warmup_epochs', default=10,    type=int)

    # ── Early stopping ──────────────────────────────────────────────────
    parser.add_argument('--early_stop',          action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stop_patience', default=20,  type=int,
                        help='Epochs without improvement before stopping')
    parser.add_argument('--early_stop_delta',    default=1e-4, type=float,
                        help='Minimum improvement considered significant')

    # ── Validation split ─────────────────────────────────────────────────
    parser.add_argument('--val_split', default=0.2, type=float,
                        help='Fraction of training data held out as a validation split '
                             '(e.g. 0.1 = 10%%). When > 0 the early-stopping signal '
                             'and logged val_loss come from this split instead of the '
                             'training loss. The existing --val_dataset is unaffected '
                             'and is still used for the standalone --eval mode.')

    # ── Mixed precision ─────────────────────────────────────────────────
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (FP16) training')

    # ── Modes ───────────────────────────────────────────────────────────
    parser.add_argument('--train',     action='store_true')
    parser.add_argument('--eval',      action='store_true')
    parser.add_argument('--finetune',  action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--force_lr',  action='store_true',
                        help='Override LR after a resume')

    # ── Chemins ─────────────────────────────────────────────────────────
    parser.add_argument('--code_dir',       type=str, default='.')
    parser.add_argument('--output_dir',     default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume',         default='',
                        help='resume from checkpoint')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='path to pretrained model; only mask head will be trained')

    # ── Logs ────────────────────────────────────────────────────────────
    parser.add_argument('--print_freq', type=int, default=10)

    # ── Données ─────────────────────────────────────────────────────────
    parser.add_argument('--bins',          type=int, default=1000)
    parser.add_argument('--chars',         type=str,
                        default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument('--padding_bins',  type=int,   default=0)
    parser.add_argument('--num_box',       type=int,   default=60)
    parser.add_argument('--pts_key',       type=str,   default='center_pts')
    parser.add_argument('--no_known_char', type=int,   default=95)
    parser.add_argument('--pad_rec_index', type=int,   default=96)
    parser.add_argument('--pad_rec',       action='store_true')
    parser.add_argument('--dict_name',     type=str, default='en_US.dic')
    parser.add_argument('--use_dict',      action='store_true')

    # ── Augmentations ───────────────────────────────────────────────────
    parser.add_argument('--max_size_train',   type=int,   default=800  )
    parser.add_argument('--min_size_train',   type=int,   nargs='+',
                        default=[448, 512, 576, 640]) #[512, 640, 672, 704, 736, 768, 800, 832, 864, 896])
    parser.add_argument('--max_size_test',    type=int,   default=960        )
    parser.add_argument('--min_size_test',    type=int,   default=640        )
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

    # ── Backbone ResNet ──────────────────────────────────────────────────
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help='name of the convolutional backbone to use')
    parser.add_argument('--dilation', action='store_true',
                        help='replace stride with dilation in the last conv block (DC5)')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'),
                        help='type of positional embedding on top of image features')

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
    parser.add_argument('--num_queries',      default=60,  type=int)
    parser.add_argument('--pre_norm',         action='store_true')
    parser.add_argument('--transformer_type', default='vanilla', type=str)

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
    parser.add_argument('--world_size', default=1,       type=int)
    parser.add_argument('--dist_url',   default='env://')
    parser.add_argument('--local_rank', default=0,       type=int)

    return parser


def build_lr_schedule(args):
    """
    Same strategy as main_original.py: linear warmup, then linear decay
    (high→low over post-warmup epochs, reversed so LR decreases over time).
    """
    if args.warmup_epochs > 0:
        warmup_lr = [
            args.warmup_min_lr
            + (args.lr - args.warmup_min_lr) * i / args.warmup_epochs
            for i in range(args.warmup_epochs)
        ]
    else:
        warmup_lr = []
    decay_epochs = args.epochs - args.warmup_epochs
    decay_lr = [
        max(i * args.lr / args.epochs, args.min_lr) for i in range(decay_epochs)
    ]
    decay_lr.reverse()
    return warmup_lr + decay_lr


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────
class EarlyStopping:
    """Stop training if loss does not improve by at least `delta`
    for `patience` consecutive epochs."""

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
            print(f"EarlyStopping: {self.counter}/{self.patience} "
                  f"(best={self.best_loss:.4f}, current={loss:.4f})")
            if self.counter >= self.patience:
                self.stop = True
                print("Early stopping triggered!")

    @property
    def improved(self):
        return self.counter == 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'delta': self.delta,
            'best_loss': self.best_loss,
            'counter': self.counter,
            'stop': self.stop,
        }

    def load_state_dict(self, state):
        self.patience = state.get('patience', self.patience)
        self.delta = state.get('delta', self.delta)
        self.best_loss = state.get('best_loss', self.best_loss)
        self.counter = state.get('counter', self.counter)
        self.stop = state.get('stop', self.stop)


# ─────────────────────────────────────────────
# Checkpoint saving
# ─────────────────────────────────────────────
def save_checkpoint(output_dir, model_without_ddp, optimizer, lr_scheduler,
                    epoch, args, save_full=False, is_best=False, early_stopping=None):
    """
    Saves:
    - checkpoint.pth          : latest checkpoint (overwritten every epoch)
    - checkpoint{epoch}.pth   : full checkpoint every 10 epochs
    - model_epoch{epoch}.pt   : weights-only every 10 epochs
    - best_model.pt           : best model according to early stopping
    """
    checkpoint = {
        'model':        model_without_ddp.state_dict(),
        'optimizer':    optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch':        epoch,
        'args':         args,
    }
    if early_stopping is not None:
        checkpoint['early_stopping'] = early_stopping.state_dict()

    utils.save_on_master(checkpoint, output_dir / 'checkpoint.pth')

    if save_full:
        utils.save_on_master(
            checkpoint,
            output_dir / f'checkpoint{epoch:04d}.pth'
        )
        utils.save_on_master(
            model_without_ddp.state_dict(),
            output_dir / f'model_epoch{epoch:04d}.pt'
        )
        print(f"Checkpoint saved: epoch {epoch:04d}")

    if is_best:
        utils.save_on_master(
            model_without_ddp.state_dict(),
            output_dir / 'best_model.pt'
        )
        print(f"Best model saved: best_model.pt (epoch {epoch})")


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
    print(f'Number of trainable parameters: {n_parameters:,}')
    print(f'Using device: {device}')

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

    # ── Mixed precision scaler ───────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if scaler is not None:
        print("Mixed precision training enabled (AMP)")

    # ── StepLR (same as main_original; per-epoch LR comes from build_lr_schedule) ──
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # ── Datasets ────────────────────────────────────────────────────────
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val   = build_dataset(image_set='val', args=args) if args.val_dataset else None

    # ── Optional validation split carved out of training data ───────────
    # When --val_split > 0 a fixed-seed random subset of the training data
    # is held out.  The split inherits the training transforms (random
    # augmentation), which adds some variance to the val loss but keeps the
    # pipeline identical to training and requires no extra annotation files.
    # The existing --val_dataset / data_loader_val is left untouched and
    # continues to be used in standalone --eval mode.
    dataset_val_loss = None
    if args.val_split > 0.0:
        n_total = len(dataset_train)
        n_val   = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(args.seed)
        dataset_train, dataset_val_loss = torch.utils.data.random_split(
            dataset_train, [n_train, n_val], generator=g
        )
        print(
            f"Val split: {n_val}/{n_total} samples held out for validation loss "
            f"({args.val_split*100:.1f}%), {n_train} remain for training."
        )

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

    data_loader_val_loss = DataLoader(
        dataset_val_loss,
        batch_size  = args.batch_size,
        sampler     = torch.utils.data.SequentialSampler(dataset_val_loss),
        drop_last   = False,
        collate_fn  = utils.collate_fn(args),
        num_workers = args.num_workers,
    ) if dataset_val_loss is not None else None

    # ── Poids figés ─────────────────────────────────────────────────────
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # ── Resume ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    resumed_early_stopping_state = None
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
                and not args.finetune
                and 'optimizer' in checkpoint
                and 'epoch' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {args.start_epoch}")
            if 'early_stopping' in checkpoint:
                resumed_early_stopping_state = checkpoint['early_stopping']

        if args.force_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.lr_backbone
            print(f"LR overridden → transformer: {args.lr}  backbone: {args.lr_backbone}")

    # ── Évaluation seule ────────────────────────────────────────────────
    if args.eval:
        if dataset_val is None:
            print('Validation dataset not found!')
            return
        evaluate(
            model, criterion,
            data_loader_val, device,
            args.output_dir, args.chars,
            args.start_index, args.category_start_index,
            args.visualize, args.max_length,
        )
        return

    # ── Entraînement ────────────────────────────────────────────────────
    print("Start training")
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
    if early_stopping is not None and resumed_early_stopping_state is not None:
        early_stopping.load_state_dict(resumed_early_stopping_state)
        print(
            "Resumed early stopping state: "
            f"best_loss={early_stopping.best_loss}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}"
        )

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
            scaler=scaler,
        )
        lr_scheduler.step()

        # ── Validation loss (teacher-forcing, no grad) ───────────────────
        val_stats = None
        if data_loader_val_loss is not None:
            val_stats = validate_loss(
                model, criterion,
                data_loader_val_loss, device,
                epoch, args.max_length,
            )

        # Early stopping monitors val loss when a split is active,
        # otherwise falls back to training loss.
        current_loss = (
            val_stats['loss'] if val_stats is not None
            else train_stats['loss']
        )

        # ── Early stopping check ─────────────────────────────────────────
        if early_stopping is not None:
            early_stopping.step(current_loss)
            is_best = early_stopping.improved
        else:
            is_best = False

        # ── Sauvegarde checkpoint ────────────────────────────────────────
        if args.output_dir:
            save_full = (
                (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0
            )
            save_checkpoint(
                output_dir, model_without_ddp, optimizer, lr_scheduler,
                epoch, args,
                save_full = save_full,
                is_best   = is_best,
                early_stopping = early_stopping,
            )

        # ── Log ─────────────────────────────────────────────────────────
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **(  {f'val_{k}': v for k, v in val_stats.items()}
                 if val_stats is not None else {}),
            'epoch':        epoch,
            'n_parameters': n_parameters,
        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + '\n')

        # ── Early stop break ─────────────────────────────────────────────
        if early_stopping is not None and early_stopping.stop:
            print(f"Training stopped at epoch {epoch} by early stopping.")
            break

    total_time     = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'SPTSv2-ResNet', parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)