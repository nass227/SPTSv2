"""
Standalone ICDAR2019 evaluation for SPTSv2.

Reads GT directly from raw .txt files (no charset confusion).
Runs the model image-by-image using predict.py-style decoding.
Sweeps confidence thresholds and writes a full report.

Usage:
    python eval_icdar2019.py ^
        --resume results/finetune_ic19/checkpoint0049.pth ^
        --test_imgs Data/ICDAR2019/test_imgs ^
        --test_gt  Data/ICDAR2019/test_gt ^
        --output_dir results/eval_ic19 ^
        --allowed_scripts Latin,Symbols,None
"""
import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from shapely.geometry import Point, LineString


import datasets.sptsv2_transforms as T
import util.misc_sptsv2 as utils
from models import build_model
from util.data import process_args


# ---------------------------------------------------------------------------
# GT reading  (directly from raw ICDAR2019 .txt, no COCO JSON involved)
# ---------------------------------------------------------------------------

def poly_center(poly_pts):
    poly_pts = np.array(poly_pts).reshape(-1, 2)
    num_points = poly_pts.shape[0]
    line1 = LineString(poly_pts[int(num_points/2):])
    line2 = LineString(poly_pts[:int(num_points/2)])
    mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
    mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
    return (mid_pt1 + mid_pt2) / 2

def read_icdar2019_gt(gt_dir, allowed_scripts=None):
    """
    Returns dict  filename_stem -> list of {x, y, text, script, dontcare}
    where x,y is the quad center in original pixel coords.
    """
    gt_dir = Path(gt_dir)
    gt_dict = {}
    for gt_path in sorted(gt_dir.glob("*.txt")):
        stem = gt_path.stem
        items = []
        with open(gt_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = list(csv.reader([line]))[0]
                    if len(parts) < 10:
                        continue
                    coords = [float(v) for v in parts[:8]]
                    script = parts[8].strip() if len(parts) > 8 else ""
                    text = parts[9].strip() if len(parts) > 9 else ""
                except Exception:
                    continue

                dontcare = (text == "" or text == "###")
                if allowed_scripts and script not in allowed_scripts:
                    dontcare = True

                cx, cy = poly_center(coords)
                items.append({
                    "x": cx, "y": cy,
                    "text": text, "script": script,
                    "dontcare": dontcare,
                })
        gt_dict[stem] = items
    return gt_dict


# ---------------------------------------------------------------------------
# Model inference  (same decoding as predict.py)
# ---------------------------------------------------------------------------

def build_transform(args):
    return T.Compose([
        T.RandomResize([args.min_size_test], args.max_size_test),
        T.ToTensor(),
        T.Normalize(None, None),
    ])


def decode_text(token_seq, chars, category_start_index):
    """Exact same logic as predict.py lines 130-137."""
    text = ""
    for c in token_seq:
        c_val = c.item() if torch.is_tensor(c) else c
        if category_start_index <= c_val < category_start_index + len(chars):
            text += chars[c_val - category_start_index]
        else:
            break
    return text


@torch.no_grad()
def run_inference(model, args, image_dir, gt_dict, transform):
    """Run model on every test image, return list of predictions."""
    device = torch.device(args.device)
    chars = list(args.chars)
    pred_length = args.max_length + 2

    all_preds = []
    total_empty = 0
    total_cand = 0
    infer_times = []

    image_dir = Path(image_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]

    image_files = []
    for ext in exts:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    printed = 0
    for img_path in tqdm(image_files, desc="Inference"):
        stem = img_path.stem
        if stem not in gt_dict:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        w_ori, h_ori = pil_img.size

        img_t, _ = transform(pil_img, None)
        c, h, w = img_t.shape
        img_t = img_t.view(1, c, h, w).to(device)

        seq = torch.ones(1, 1, dtype=torch.long, device=device) * args.start_index

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        output = model(img_t, seq, seq, text_length=args.max_length)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        infer_times.append(t1 - t0)

        if output is None:
            continue

        outputs, values, _ = output
        out = outputs[0]
        val = values[0]
        n = out.shape[0] // pred_length

        for i in range(n):
            s = i * pred_length
            conf = float(val[s:s + pred_length].mean().item())
            val_xy = float(val[s:s + 2].mean().item())
            px = float(out[s].item()) * (w_ori / 1000.0)
            py = float(out[s + 1].item()) * (h_ori / 1000.0)
            token_seq = out[s + 2:s + pred_length]
            rec = decode_text(token_seq, chars, args.category_start_index)
            rec_len = len(rec)
            val_rec = float(val[s + 2:s + 2 + rec_len].mean().item()) if rec_len > 0 else 0.0

            total_cand += 1
            if not rec:
                total_empty += 1

            if printed < 30:
                print(
                    f"[cand {printed:03d}] {stem} conf={conf:.4f} "
                    f"val_xy={val_xy:.4f} val_rec={val_rec:.4f} "
                    f"xy=({px:.1f},{py:.1f}) empty={not rec} text='{rec}' "
                    f"tokens={token_seq[:6].tolist()}"
                )
                printed += 1

            if rec:
                all_preds.append({
                    "file_stem": stem,
                    "x": px, "y": py,
                    "rec": rec,
                    "score": conf,
                    "val_xy": val_xy,
                    "val_rec": val_rec,
                })

    stats = {
        "total_candidates": total_cand,
        "total_empty": total_empty,
        "total_kept": len(all_preds),
        "avg_infer_s": float(np.mean(infer_times)) if infer_times else 0.0,
        "num_images": len(infer_times),
    }
    return all_preds, stats


# ---------------------------------------------------------------------------
# Evaluation  (detection + end-to-end spotting, like eval_ic15.py)
# ---------------------------------------------------------------------------

def _prf(ntp, ngt, ndet):
    if ndet == 0 or ntp == 0:
        return 0.0, 0.0, 0.0
    p = ntp / ndet
    r = ntp / ngt
    f = 2 * p * r / (p + r)
    return p, r, f


def evaluate_at_threshold(preds, gt_dict, conf_thr, conf_rec_thr=0.0,
                          case_sensitive=False):
    """
    Returns separate Detection and End-to-End (spotting) P/R/F1.

    Filtering pipeline (mirrors eval_ic15.py):
      1. Primary threshold on mean sequence confidence (``score >= conf_thr``).
      2. Hard per-character recognition threshold
         (``val_rec >= conf_rec_thr``).

    Detection TP  : closest non-dontcare GT matched by location only.
    End-to-end TP : same location match AND exact text match.
    """
    gt_copy = copy.deepcopy(gt_dict)

    ngt = sum(1 for items in gt_copy.values()
              for it in items if not it["dontcare"])

    filtered = [p for p in preds
                if p["score"] >= conf_thr
                and p.get("val_rec", 1.0) >= conf_rec_thr]
    filtered.sort(key=lambda x: -x["score"])

    matched_det = {stem: [False] * len(items)
                   for stem, items in gt_copy.items()}
    matched_e2e = {stem: [False] * len(items)
                   for stem, items in gt_copy.items()}

    ndet = 0
    ntp_det = 0
    ntp_e2e = 0

    for pred in filtered:
        stem = pred["file_stem"]
        if stem not in gt_copy:
            ndet += 1
            continue

        gt_items = gt_copy[stem]
        if not gt_items:
            ndet += 1
            continue

        dists = [math.hypot(pred["x"] - g["x"], pred["y"] - g["y"])
                 for g in gt_items]
        idx = int(np.argmin(dists))
        g = gt_items[idx]

        if g["dontcare"] or g["text"] == "" or g["text"] == "###":
            continue

        ndet += 1

        if not matched_det[stem][idx]:
            matched_det[stem][idx] = True
            ntp_det += 1

        pred_t = pred["rec"] if case_sensitive else pred["rec"].upper()
        gt_t = g["text"] if case_sensitive else g["text"].upper()
        if pred_t == gt_t and not matched_e2e[stem][idx]:
            matched_e2e[stem][idx] = True
            ntp_e2e += 1

    p_det, r_det, f_det = _prf(ntp_det, ngt, ndet)
    p_e2e, r_e2e, f_e2e = _prf(ntp_e2e, ngt, ndet)

    return {
        "det":  {"p": p_det, "r": r_det, "f1": f_det,
                 "tp": ntp_det, "ngt": ngt, "ndet": ndet},
        "e2e":  {"p": p_e2e, "r": r_e2e, "f1": f_e2e,
                 "tp": ntp_e2e, "ngt": ngt, "ndet": ndet},
    }


def sweep_thresholds(preds, gt_dict, conf_rec_thr=0.0,
                     lo=0.3, hi=0.95, step=0.01):
    """Sweep primary confidence thresholds; return rows + best for each task."""
    thresholds = np.arange(lo, hi + step, step)
    rows = []
    for thr in thresholds:
        res = evaluate_at_threshold(preds, gt_dict, thr,
                                    conf_rec_thr=conf_rec_thr)
        rows.append({
            "threshold": float(thr),
            **{f"det_{k}": v for k, v in res["det"].items()},
            **{f"e2e_{k}": v for k, v in res["e2e"].items()},
        })
    best_det = max(rows, key=lambda x: x["det_f1"])
    best_e2e = max(rows, key=lambda x: x["e2e_f1"])
    return rows, best_det, best_e2e


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _best_row_section(lines, label, rows, key_p, key_r, key_f, key_tp, key_ndet, best):
    lines.append(f"--- Best {label} Result ---")
    lines.append(f"conf threshold   : {best['threshold']:.3f}")
    lines.append(f"precision        : {best[key_p]:.4f}")
    lines.append(f"recall           : {best[key_r]:.4f}")
    lines.append(f"F1 (hmean)       : {best[key_f]:.4f}")
    lines.append(f"true positives   : {best[key_tp]}")
    lines.append(f"detections       : {best[key_ndet]}")
    lines.append(f"GT (non-dontcare): {best['det_ngt']}")
    lines.append("")
    lines.append(f"--- {label} Threshold Sweep ---")
    lines.append(f"{'thr':>6s}  {'prec':>7s}  {'rec':>7s}  {'f1':>7s}  {'tp':>5s}  {'ndet':>6s}")
    SHOW = {0.3, 0.5, 0.55, 0.7, 0.8, 0.9}
    for r in rows:
        if r[key_f] > 0 or round(r["threshold"], 2) in SHOW:
            lines.append(
                f"{r['threshold']:6.3f}  {r[key_p]:7.4f}  {r[key_r]:7.4f}  "
                f"{r[key_f]:7.4f}  {r[key_tp]:5d}  {r[key_ndet]:6d}"
            )
    lines.append("")


def write_report(output_dir, preds, stats, rows, best_det, best_e2e, gt_dict, args):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("  ICDAR2019 Evaluation Report  -  SPTSv2")
    lines.append("=" * 60)
    lines.append(f"checkpoint       : {args.resume}")
    lines.append(f"test images      : {args.test_imgs}")
    lines.append(f"test GT          : {args.test_gt}")
    lines.append(f"allowed scripts  : {args.allowed_scripts}")
    lines.append(f"chars (len={len(args.chars)}): {args.chars[:40]}...")
    lines.append(f"pad_rec          : {args.pad_rec}")
    lines.append(f"max_length       : {args.max_length}")
    lines.append(f"conf_rec_thr     : {args.conf_rec:.3f}")
    lines.append(f"category_start_index: {args.category_start_index}")
    lines.append(f"start_index      : {args.start_index}")
    lines.append(f"end_index        : {args.end_index}")
    lines.append("")
    lines.append("--- Inference Stats ---")
    lines.append(f"images processed : {stats['num_images']}")
    lines.append(f"avg inference (s): {stats['avg_infer_s']:.4f}")
    lines.append(f"total candidates : {stats['total_candidates']}")
    lines.append(f"empty strings    : {stats['total_empty']} "
                 f"({100*stats['total_empty']/max(stats['total_candidates'],1):.1f}%)")
    lines.append(f"non-empty kept   : {stats['total_kept']}")

    total_gt = sum(1 for items in gt_dict.values() for it in items if not it["dontcare"])
    lines.append(f"GT instances     : {total_gt}")
    lines.append("")

    # --- Detection ---
    _best_row_section(lines, "Detection",
                      rows,
                      "det_p", "det_r", "det_f1", "det_tp", "det_ndet",
                      best_det)

    # --- End-to-End Spotting ---
    _best_row_section(lines, "End-to-End Spotting",
                      rows,
                      "e2e_p", "e2e_r", "e2e_f1", "e2e_tp", "e2e_ndet",
                      best_e2e)

    lines.append("--- Sample Predictions (first 20 kept) ---")
    for p in preds[:20]:
        lines.append(
            f"  {p['file_stem']} | '{p['rec']}' | score {p['score']:.3f} "
            f"val_rec={p.get('val_rec', float('nan')):.3f} "
            f"| ({p['x']:.1f}, {p['y']:.1f})"
        )
    lines.append("")

    sample_stems = list(gt_dict.keys())[:5]
    lines.append("--- Sample GT (first 5 images) ---")
    for stem in sample_stems:
        for g in gt_dict[stem][:5]:
            dc = " [DONTCARE]" if g["dontcare"] else ""
            lines.append(
                f"  {stem} | '{g['text']}' | ({g['x']:.1f}, {g['y']:.1f}) "
                f"| {g['script']}{dc}"
            )
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)

    report_path = out / "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    json_path = out / "predictions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    sweep_path = out / "threshold_sweep.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(report)
    print(f"\nSaved report   : {report_path}")
    print(f"Saved preds    : {json_path}")
    print(f"Saved sweep    : {sweep_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser("ICDAR2019 SPTSv2 Evaluation")
    parser.add_argument("--resume", required=True, help="Checkpoint .pth path")
    parser.add_argument("--test_imgs", default="Data/ICDAR2019/test_imgs")
    parser.add_argument("--test_gt", default="Data/ICDAR2019/test_gt")
    parser.add_argument("--output_dir", default="results/eval_ic19")
    parser.add_argument("--allowed_scripts", type=str, default="Latin,Symbols,None",
                        help="Comma-separated scripts to evaluate on (empty=all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # model args (must match checkpoint)
    parser.add_argument("--lr_backbone", type=float, default=0)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine")
    parser.add_argument("--enc_layers", type=int, default=6)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--depths", type=int, default=6)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--masks", action="store_true")
    # data / vocab args
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--chars", type=str,
                        default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
    parser.add_argument("--padding_bins", type=int, default=0)
    parser.add_argument("--pad_rec", action="store_true")
    parser.add_argument("--pad_rec_index", type=int, default=96)
    parser.add_argument("--no_known_char", type=int, default=95)
    parser.add_argument("--max_length", type=int, default=25)
    parser.add_argument("--num_box", type=int, default=60)
    parser.add_argument("--obj_num", type=int, default=60)
    parser.add_argument("--pts_key", default="center_pts")
    parser.add_argument("--max_size_test", type=int, default=1824)
    parser.add_argument("--min_size_test", type=int, default=1000)
    # evaluation thresholds
    parser.add_argument("--conf_rec", type=float, default=0.62,
                        help="Per-character recognition hard threshold (like eval_ic15)")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = process_args(args)

    print(f"Token indices: category_start={args.category_start_index}, "
          f"end={args.end_index}, start={args.start_index}")

    # Parse allowed scripts
    if args.allowed_scripts.strip():
        allowed = {s.strip() for s in args.allowed_scripts.split(",") if s.strip()}
    else:
        allowed = None
    print(f"Allowed scripts: {allowed or 'ALL'}")

    # Read GT
    print("Reading GT...")
    gt_dict = read_icdar2019_gt(args.test_gt, allowed)
    total_gt = sum(1 for items in gt_dict.values() for it in items if not it["dontcare"])
    total_dc = sum(1 for items in gt_dict.values() for it in items if it["dontcare"])
    print(f"GT: {len(gt_dict)} images, {total_gt} eval instances, {total_dc} dontcare")

    # Build model
    print("Building model...")
    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {args.resume}")

    # Run inference
    print("Running inference...")
    transform = build_transform(args)
    preds, stats = run_inference(model, args, args.test_imgs, gt_dict, transform)
    preds.sort(key=lambda x: -x["score"])

    # Evaluate
    print(f"Evaluating (threshold sweep, conf_rec_thr={args.conf_rec:.3f})...")
    rows, best_det, best_e2e = sweep_thresholds(
        preds, gt_dict, conf_rec_thr=args.conf_rec, lo=0.3, hi=0.95)

    # Write report
    write_report(args.output_dir, preds, stats, rows, best_det, best_e2e,
                 gt_dict, args)


if __name__ == "__main__":
    main()
