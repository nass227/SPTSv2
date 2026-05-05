"""
Standalone ICDAR2019 evaluation for SPTSv2.

Reads GT from a COCO-style JSON produced by build_annotations_json.py
(--rec-string recommended but not required).  Runs the model image-by-image,
sweeps confidence thresholds, and writes a full report including Detection,
End-to-End Spotting, AP, ECE, and recognition quality.

Expected JSON schema (output of build_annotations_json.py):
  {
    "images":      [{"id", "file_name", "width", "height"}, ...],
    "annotations": [{"id", "image_id", "category_id",
                      "bbox": [x, y, w, h],  <- COCO top-left + size
                      "area", "iscrowd",
                      "bezier_pts", "rec": [int x max_length],
                      "rec_string": "..."    <- optional plain text
                    }, ...],
    "categories":  [{"id": 1, "name": "text"}]
  }

  * bbox format  : [x_topleft, y_topleft, width, height]
  * rec encoding : charset indices 0..(len(chars)-1); pad = len(chars)
  * rec_string   : plain UTF-8 text (preferred; avoids charset alignment issues)

Usage:
    python eval_icdar2019.py ^
        --resume results/finetune_ic19/checkpoint0049.pth ^
        --test_imgs Data/ICDAR2019/test_imgs ^
        --gt_json  Data/ICDAR2019/test_annotations.json ^
        --output_dir results/eval_ic19

Notes:
  - Build the JSON with --rec-string to embed plain text and avoid any
    charset-alignment issues between build_annotations_json.py and this script.
  - The default --chars matches build_annotations_json.py / main.py exactly
    (94 printable ASCII starting from '!', plus 34 French accented chars).
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


import datasets.sptsv2_transforms as T
import util.misc_sptsv2 as utils
from models import build_model
from util.data import process_args


# ---------------------------------------------------------------------------
# GT reading — COCO-style JSON  (build_annotations_json.py / pipeline output)
# ---------------------------------------------------------------------------

def _decode_rec(rec_ints, chars):
    """
    Decode a list of integer token indices back to a text string.
    Any index >= len(chars) (PAD / OOV / special) terminates decoding.
    """
    pad_idx = len(chars)
    text = ""
    for i in rec_ints:
        if i >= pad_idx:
            break
        text += chars[i]
    return text


def read_coco_gt(json_path, decode_chars=None):
    """
    Read GT from a COCO-style JSON produced by build_annotations_json.py.

    Returns dict  stem -> list of {x, y, text, script, dontcare}

    JSON format expected:
      images      : [{"id", "file_name", "width", "height"}, ...]
      annotations : [{"id", "image_id", "category_id",
                       "bbox": [x_topleft, y_topleft, width, height],
                       "rec":  [int, ...],        <- charset indices, padded
                       "rec_string": "..."         <- optional plain text
                      }, ...]

    Text is sourced from (in priority order):
      1. ``rec_string`` field  — present when JSON was built with --rec-string
      2. Decoded ``rec`` field — decode_chars must match the charset used at build

    Dontcare rules (ICDAR convention):
      - annotation text == "" or "###"  → dontcare
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_list = data.get("images", [])
    anns_list   = data.get("annotations", [])

    if not images_list:
        raise ValueError(f"COCO JSON has no 'images' entries: {json_path}")
    if not anns_list:
        raise ValueError(f"COCO JSON has no 'annotations' entries: {json_path}")

    # Validate top-level keys
    first_ann = anns_list[0]
    missing = [k for k in ("image_id", "bbox") if k not in first_ann]
    if missing:
        raise ValueError(
            f"First annotation is missing required fields {missing}. "
            f"Check that the JSON was produced by build_annotations_json.py."
        )

    # Build image_id -> stem mapping
    id_to_stem = {
        img["id"]: Path(img["file_name"]).stem
        for img in images_list
    }

    has_rec_string = any("rec_string" in ann for ann in anns_list)
    if not has_rec_string:
        if decode_chars is None:
            raise ValueError(
                "COCO JSON has no 'rec_string' fields and --chars was not supplied.\n"
                "Either:\n"
                "  (a) rebuild the JSON with --rec-string to embed plain text, or\n"
                "  (b) ensure --chars matches the charset used during JSON build."
            )
        print(
            f"[INFO] JSON has no 'rec_string'; decoding 'rec' indices with "
            f"provided charset (len={len(decode_chars)})."
        )
    else:
        print(f"[INFO] Using 'rec_string' field for GT text decoding.")

    gt_dict: dict = {}
    n_dontcare = 0
    for ann in anns_list:
        stem = id_to_stem.get(ann["image_id"])
        if stem is None:
            continue

        # Decode text ----------------------------------------------------------
        if "rec_string" in ann:
            text = ann["rec_string"]
        else:
            text = _decode_rec(ann["rec"], decode_chars)

        # Centre from COCO bbox [x_topleft, y_topleft, width, height] ----------
        x1, y1, bw, bh = ann["bbox"]
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        dontcare = (text.strip() == "" or text.strip() == "###")
        if dontcare:
            n_dontcare += 1

        gt_dict.setdefault(stem, []).append({
            "x": cx, "y": cy,
            "text": text,
            "script": ann.get("script", ""),
            "dontcare": dontcare,
        })

    n_eval = sum(1 for items in gt_dict.values()
                 for it in items if not it["dontcare"])
    print(
        f"[INFO] GT loaded: {len(gt_dict)} images | "
        f"{n_eval} eval instances | {n_dontcare} dontcare"
    )
    return gt_dict


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def build_transform(args):
    return T.Compose([
        T.RandomResize([args.min_size_test], args.max_size_test),
        T.ToTensor(),
        T.Normalize(None, None),
    ])


def decode_text(token_seq, chars, category_start_index):
    """Exact same logic as predict.py."""
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
    """Run model on every test image that has a GT entry; return predictions."""
    device = torch.device(args.device)
    chars = list(args.chars)
    pred_length = args.max_length + 2

    all_preds = []
    total_empty = 0
    total_cand = 0
    infer_times = []

    image_dir = Path(image_dir)
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]

    # Resolve to absolute paths and deduplicate (safe on case-insensitive FS)
    image_files = sorted({
        p.resolve()
        for ext in exts
        for p in image_dir.glob(f"*{ext}")
    })

    no_gt_count = 0
    printed = 0
    for img_path in tqdm(image_files, desc="Inference"):
        stem = img_path.stem
        if stem not in gt_dict:
            no_gt_count += 1
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
            conf     = float(val[s:s + pred_length].mean().item())
            val_xy   = float(val[s:s + 2].mean().item())
            px       = float(out[s].item())     * (w_ori / 1000.0)
            py       = float(out[s + 1].item()) * (h_ori / 1000.0)
            token_seq = out[s + 2:s + pred_length]
            rec       = decode_text(token_seq, chars, args.category_start_index)
            rec_len   = len(rec)
            val_rec   = float(val[s + 2:s + 2 + rec_len].mean().item()) if rec_len > 0 else 0.0

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

    if no_gt_count > 0:
        print(f"[WARN] {no_gt_count} image(s) had no matching GT entry and were skipped. "
              f"Check that image filenames match GT keys (stems must match exactly).")

    stats = {
        "total_candidates": total_cand,
        "total_empty":      total_empty,
        "total_kept":       len(all_preds),
        "avg_infer_s":      float(np.mean(infer_times)) if infer_times else 0.0,
        "num_images":       len(infer_times),
    }
    return all_preds, stats


# ---------------------------------------------------------------------------
# Evaluation — detection + end-to-end spotting
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

    Filtering pipeline:
      1. Primary threshold on mean sequence confidence (score >= conf_thr).
      2. Hard per-character recognition threshold   (val_rec >= conf_rec_thr).

    Detection TP  : closest non-dontcare GT matched by location only.
    End-to-End TP : same location match AND exact text match.

    Predictions whose nearest GT is a dontcare region are absorbed (not
    counted as either TP or FP) — standard ICDAR convention.
    Predictions on images absent from gt_dict are counted as FP detections.
    """
    gt_copy = copy.deepcopy(gt_dict)

    ngt = sum(1 for items in gt_copy.values()
              for it in items if not it["dontcare"])

    filtered = [p for p in preds
                if p["score"] >= conf_thr
                and p.get("val_rec", 0.0) >= conf_rec_thr]
    filtered.sort(key=lambda x: -x["score"])

    matched_det = {stem: [False] * len(items)
                   for stem, items in gt_copy.items()}
    matched_e2e = {stem: [False] * len(items)
                   for stem, items in gt_copy.items()}

    ndet    = 0
    ntp_det = 0
    ntp_e2e = 0

    for pred in filtered:
        stem = pred["file_stem"]

        # Image not in GT at all → counts as a false positive detection
        if stem not in gt_copy or not gt_copy[stem]:
            ndet += 1
            continue

        gt_items = gt_copy[stem]
        dists = [math.hypot(pred["x"] - g["x"], pred["y"] - g["y"])
                 for g in gt_items]
        idx = int(np.argmin(dists))
        g   = gt_items[idx]

        # Nearest GT is dontcare → prediction is absorbed, not penalised
        if g["dontcare"]:
            continue

        ndet += 1

        if not matched_det[stem][idx]:
            matched_det[stem][idx] = True
            ntp_det += 1

        pred_t = pred["rec"] if case_sensitive else pred["rec"].upper()
        gt_t   = g["text"]  if case_sensitive else g["text"].upper()
        if pred_t == gt_t and not matched_e2e[stem][idx]:
            matched_e2e[stem][idx] = True
            ntp_e2e += 1

    p_det, r_det, f_det = _prf(ntp_det, ngt, ndet)
    p_e2e, r_e2e, f_e2e = _prf(ntp_e2e, ngt, ndet)

    return {
        "det": {"p": p_det, "r": r_det, "f1": f_det,
                "tp": ntp_det, "ngt": ngt, "ndet": ndet},
        "e2e": {"p": p_e2e, "r": r_e2e, "f1": f_e2e,
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
# Rich analytics: AP, ECE, recognition quality, error breakdown
# ---------------------------------------------------------------------------

def analyze_predictions(preds, gt_dict, conf_rec_thr=0.0, case_sensitive=False):
    """
    Single greedy-matching pass with NO score threshold (only conf_rec_thr
    applied).  Assigns is_det_tp / is_e2e_tp to every prediction.

    Handling of special cases:
      • Prediction on an image absent from gt_dict → FP (is_det_tp=False)
      • Prediction whose nearest GT is dontcare    → absorbed, excluded from
        labeled (ICDAR convention; same as evaluate_at_threshold)

    The returned list is sorted by descending score — ready for AP / PR-curve
    computation and for ECE / recognition-quality analysis.

    Returns (labeled, ngt).
    """
    gt_copy = copy.deepcopy(gt_dict)
    ngt = sum(1 for items in gt_copy.values()
              for it in items if not it["dontcare"])

    # Apply only the recognition hard-threshold (no score threshold for AP)
    filtered = [p for p in preds if p.get("val_rec", 0.0) >= conf_rec_thr]
    filtered.sort(key=lambda x: -x["score"])

    matched_det = {stem: [False] * len(items) for stem, items in gt_copy.items()}
    matched_e2e = {stem: [False] * len(items) for stem, items in gt_copy.items()}

    labeled = []
    for pred in filtered:
        stem = pred["file_stem"]

        # Image absent from GT → false positive; add to labeled so AP is correct
        if stem not in gt_copy or not gt_copy[stem]:
            labeled.append({
                "score":     pred["score"],
                "val_rec":   pred.get("val_rec", 0.0),
                "val_xy":    pred.get("val_xy", 0.0),
                "is_det_tp": False,
                "is_e2e_tp": False,
                "pred_text": pred["rec"],
                "gt_text":   "",
            })
            continue

        gt_items = gt_copy[stem]
        dists = [math.hypot(pred["x"] - g["x"], pred["y"] - g["y"])
                 for g in gt_items]
        idx = int(np.argmin(dists))
        g   = gt_items[idx]

        # Nearest GT is dontcare → prediction is absorbed (not a FP per ICDAR)
        if g["dontcare"]:
            continue

        is_det_tp = False
        is_e2e_tp = False

        if not matched_det[stem][idx]:
            matched_det[stem][idx] = True
            is_det_tp = True

        pred_t = pred["rec"] if case_sensitive else pred["rec"].upper()
        gt_t   = g["text"]  if case_sensitive else g["text"].upper()
        if pred_t == gt_t and not matched_e2e[stem][idx]:
            matched_e2e[stem][idx] = True
            is_e2e_tp = True

        labeled.append({
            "score":     pred["score"],
            "val_rec":   pred.get("val_rec", 0.0),
            "val_xy":    pred.get("val_xy", 0.0),
            "is_det_tp": is_det_tp,
            "is_e2e_tp": is_e2e_tp,
            "pred_text": pred["rec"],
            "gt_text":   g["text"],
        })

    return labeled, ngt


def compute_ap(labeled, ngt, task="det"):
    """
    Average Precision via trapezoidal AUC of the precision-recall curve.
    ``labeled`` must be sorted by descending score (guaranteed by
    ``analyze_predictions``).  Curve starts at (recall=0, precision=1).
    """
    key = "is_det_tp" if task == "det" else "is_e2e_tp"
    precisions, recalls = [1.0], [0.0]
    ntp = 0
    for i, item in enumerate(labeled):
        if item[key]:
            ntp += 1
        precisions.append(ntp / (i + 1))
        recalls.append(ntp / ngt if ngt > 0 else 0.0)
    ap = float(np.trapz(precisions, recalls))
    return ap, precisions, recalls


def compute_ece(labeled, task="det", n_bins=15):
    """
    Expected Calibration Error: measures how well ``score`` predicts accuracy.
    Bins predictions uniformly; ECE = weighted mean |conf - acc|.
    """
    key = "is_det_tp" if task == "det" else "is_e2e_tp"
    if not labeled:
        return 0.0, []
    scores  = np.array([it["score"]   for it in labeled])
    correct = np.array([int(it[key])  for it in labeled])
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    n       = len(scores)
    ece     = 0.0
    bins_out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores <= hi if i == n_bins - 1 else scores < hi)
        if not mask.any():
            continue
        avg_conf = float(scores[mask].mean())
        avg_acc  = float(correct[mask].mean())
        weight   = int(mask.sum()) / n
        ece     += weight * abs(avg_conf - avg_acc)
        bins_out.append({
            "lo": round(lo, 3), "hi": round(hi, 3),
            "count":    int(mask.sum()),
            "avg_conf": round(avg_conf, 4),
            "avg_acc":  round(avg_acc,  4),
            "gap":      round(abs(avg_conf - avg_acc), 4),
        })
    return round(ece, 6), bins_out


def compute_rec_metrics(labeled):
    """
    Recognition-specific metrics computed over Detection TPs only.

    Metrics
    -------
    exact_match   – word-level exact accuracy  (case-insensitive)
    1-NED         – 1 - mean normalised edit distance
    CRR           – Character Recall Rate = 1 - CER  (char-level TP / GT chars)
    avg_edit_dist – mean edit distance per word
    """
    det_tps = [it for it in labeled if it["is_det_tp"]]
    if not det_tps:
        return {}

    try:
        import editdistance as _ed
        _edit = lambda a, b: _ed.eval(a, b)
    except ImportError:
        def _edit(a, b):
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for ci in range(1, m + 1):
                prev, dp[0] = dp[:], ci
                for cj in range(1, n + 1):
                    dp[cj] = (prev[cj - 1] if a[ci - 1] == b[cj - 1]
                              else 1 + min(prev[cj - 1], prev[cj], dp[cj - 1]))
            return dp[n]

    total_ned      = 0.0
    total_ed       = 0
    total_chars_gt = 0
    n_exact        = 0

    for it in det_tps:
        pred = it["pred_text"].upper()
        gt   = it["gt_text"].upper()
        d = _edit(pred, gt)
        total_ed       += d
        total_ned      += d / max(len(pred), len(gt), 1)
        total_chars_gt += len(gt)
        if pred == gt:
            n_exact += 1

    n = len(det_tps)
    return {
        "n_det_tp":      n,
        "exact_match":   round(n_exact / n, 4),
        "one_minus_ned": round(1.0 - total_ned / n, 4),
        "crr":           round(1.0 - total_ed / max(total_chars_gt, 1), 4),
        "avg_edit_dist": round(total_ed / n, 3),
    }


def compute_error_analysis(labeled, conf_thr, conf_rec_thr=0.0):
    """
    At (conf_thr, conf_rec_thr), partition matched predictions into three
    mutually exclusive buckets:

      det_ok_rec_ok     – location TP  AND  text match   (ntp_e2e)
      det_ok_rec_wrong  – location TP  but  text wrong   (ntp_det - ntp_e2e)
      det_wrong         – location FP

    Note: ``is_det_tp`` / ``is_e2e_tp`` flags were assigned during the
    unthresholded greedy pass in analyze_predictions.  At high thresholds a
    small number of flags may be stale (a lower-confidence TP was assigned
    before a higher-confidence one is removed by thresholding).  This is an
    acceptable approximation; the F1-sweep table gives exact figures at any
    specific threshold.
    """
    filt     = [it for it in labeled
                if it["score"] >= conf_thr and it["val_rec"] >= conf_rec_thr]
    both      = sum(1 for it in filt if it["is_det_tp"] and it["is_e2e_tp"])
    det_no_rec = sum(1 for it in filt if it["is_det_tp"] and not it["is_e2e_tp"])
    det_fp    = sum(1 for it in filt if not it["is_det_tp"])
    n = len(filt)
    return {
        "n_total":          n,
        "det_ok_rec_ok":    both,
        "det_ok_rec_wrong": det_no_rec,
        "det_wrong":        det_fp,
    }


# ---------------------------------------------------------------------------
# PR curve helpers
# ---------------------------------------------------------------------------

def _sample_pr_at_recalls(precisions, recalls, levels=None):
    """Return (recall, precision) sampled at ``levels`` via linear interpolation."""
    if levels is None:
        levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pairs = sorted(zip(recalls, precisions))
    r_arr = np.array([p[0] for p in pairs])
    p_arr = np.array([p[1] for p in pairs])
    out = []
    for lv in levels:
        p_interp = float(np.interp(lv, r_arr, p_arr))
        out.append((round(lv, 2), round(p_interp, 4)))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _best_row_section(lines, label, rows,
                      key_p, key_r, key_f, key_tp, key_ndet, key_ngt,
                      best):
    lines.append(f"--- Best {label} Result ---")
    lines.append(f"conf threshold   : {best['threshold']:.3f}")
    lines.append(f"precision        : {best[key_p]:.4f}")
    lines.append(f"recall           : {best[key_r]:.4f}")
    lines.append(f"F1 (hmean)       : {best[key_f]:.4f}")
    lines.append(f"true positives   : {best[key_tp]}")
    lines.append(f"detections       : {best[key_ndet]}")
    lines.append(f"GT (non-dontcare): {best[key_ngt]}")
    lines.append("")
    lines.append(f"--- {label} Threshold Sweep ---")
    lines.append(f"{'thr':>6s}  {'prec':>7s}  {'rec':>7s}  {'f1':>7s}  "
                 f"{'tp':>5s}  {'ndet':>6s}")
    SHOW = {0.3, 0.5, 0.55, 0.7, 0.8, 0.9}
    for r in rows:
        if r[key_f] > 0 or round(r["threshold"], 2) in SHOW:
            lines.append(
                f"{r['threshold']:6.3f}  {r[key_p]:7.4f}  {r[key_r]:7.4f}  "
                f"{r[key_f]:7.4f}  {r[key_tp]:5d}  {r[key_ndet]:6d}"
            )
    lines.append("")


def write_report(output_dir, preds, stats, rows, best_det, best_e2e,
                 analytics, gt_dict, args):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ngt          = analytics.get("ngt", 0)
    ap_det       = analytics.get("ap_det",  0.0)
    ap_e2e       = analytics.get("ap_e2e",  0.0)
    ece_det      = analytics.get("ece_det", 0.0)
    ece_e2e      = analytics.get("ece_e2e", 0.0)
    ece_det_bins = analytics.get("ece_det_bins", [])
    ece_e2e_bins = analytics.get("ece_e2e_bins", [])
    rec_m        = analytics.get("rec_metrics", {})
    err_det      = analytics.get("err_det", {})
    err_e2e      = analytics.get("err_e2e", {})
    pr_det_sampled = analytics.get("pr_det_sampled", [])
    pr_e2e_sampled = analytics.get("pr_e2e_sampled", [])

    gt_source = getattr(args, "gt_json", None) or getattr(args, "test_gt", "")

    lines = []
    W = 68
    lines.append("=" * W)
    lines.append("  ICDAR2019 Evaluation Report  -  SPTSv2")
    lines.append("=" * W)
    lines.append(f"checkpoint       : {args.resume}")
    lines.append(f"test images      : {args.test_imgs}")
    lines.append(f"GT source        : {gt_source}")
    chars_preview = args.chars[:40].encode("ascii", "replace").decode()
    lines.append(f"chars (len={len(args.chars)}): {chars_preview}...")
    lines.append(f"pad_rec          : {args.pad_rec}")
    lines.append(f"max_length       : {args.max_length}")
    lines.append(f"conf_rec_thr     : {args.conf_rec:.3f}")
    lines.append(f"category_start_index: {args.category_start_index}")
    lines.append(f"start_index      : {args.start_index}")
    lines.append(f"end_index        : {args.end_index}")
    lines.append("")

    # Inference stats
    lines.append("--- Inference Stats ---")
    lines.append(f"images processed : {stats['num_images']}")
    lines.append(f"avg inference (s): {stats['avg_infer_s']:.4f}")
    lines.append(f"total candidates : {stats['total_candidates']}")
    lines.append(
        f"empty strings    : {stats['total_empty']} "
        f"({100 * stats['total_empty'] / max(stats['total_candidates'], 1):.1f}%)"
    )
    lines.append(f"non-empty kept   : {stats['total_kept']}")
    total_gt = sum(1 for items in gt_dict.values() for it in items if not it["dontcare"])
    lines.append(f"GT instances     : {total_gt}")
    lines.append("")

    # Detection sweep
    _best_row_section(lines, "Detection",
                      rows, "det_p", "det_r", "det_f1", "det_tp", "det_ndet", "det_ngt",
                      best_det)

    # End-to-End sweep
    _best_row_section(lines, "End-to-End Spotting",
                      rows, "e2e_p", "e2e_r", "e2e_f1", "e2e_tp", "e2e_ndet", "e2e_ngt",
                      best_e2e)

    # Average Precision
    lines.append("--- Average Precision (AP) ---")
    lines.append(f"AP  Detection    : {ap_det:.4f}")
    lines.append(f"AP  End-to-End   : {ap_e2e:.4f}")
    lines.append("")

    # PR curve (sampled)
    lines.append("--- Precision-Recall Curve (sampled at fixed recall levels) ---")
    lines.append(f"  {'Recall':>7s}  {'Det-P':>7s}  {'E2E-P':>7s}")
    pr_e2e_map = dict(pr_e2e_sampled)
    for rec_lv, p_det in pr_det_sampled:
        p_e2e = pr_e2e_map.get(rec_lv, float("nan"))
        lines.append(f"  {rec_lv:7.2f}  {p_det:7.4f}  {p_e2e:7.4f}")
    lines.append(f"  (full curve saved in analytics.json)")
    lines.append("")

    # Error analysis
    def _err_section(label, err, best_thr, ngt_val):
        lines.append(
            f"--- Error Analysis @ best {label} threshold"
            f" (thr={best_thr:.3f}, conf_rec={args.conf_rec:.3f}) ---"
        )
        n      = err.get("n_total", 0)
        both   = err.get("det_ok_rec_ok",    0)
        d_rw   = err.get("det_ok_rec_wrong", 0)
        d_fp   = err.get("det_wrong",        0)
        missed = ngt_val - (both + d_rw)
        pct    = lambda x: f"{100 * x / max(n, 1):5.1f}%"
        lines.append(f"  Total matched predictions  : {n}")
        lines.append(f"  Det correct + Rec correct  : {both:6d}  ({pct(both)}  of matched)")
        lines.append(f"  Det correct + Rec WRONG    : {d_rw:6d}  ({pct(d_rw)}  of matched)")
        lines.append(f"  Det WRONG   (FP)           : {d_fp:6d}  ({pct(d_fp)}  of matched)")
        lines.append(
            f"  Missed GT   (FN)           : {missed:6d}"
            f"  ({100 * missed / max(ngt_val, 1):5.1f}%  of GT)"
        )
        lines.append("")

    _err_section("Detection",    err_det, best_det["threshold"], ngt)
    _err_section("E2E Spotting", err_e2e, best_e2e["threshold"], ngt)

    # Recognition quality
    lines.append("--- Recognition Quality (over Detection TPs, all confidence levels) ---")
    if rec_m:
        lines.append(f"  Det TPs evaluated    : {rec_m['n_det_tp']}")
        lines.append(f"  Exact-match rate     : {rec_m['exact_match']:.4f}")
        lines.append(f"  1 - NED              : {rec_m['one_minus_ned']:.4f}"
                     f"  (1 = perfect, 0 = completely wrong)")
        lines.append(f"  Char Recall Rate     : {rec_m['crr']:.4f}"
                     f"  (1 - CER, char-level)")
        lines.append(f"  Avg edit distance    : {rec_m['avg_edit_dist']:.3f} chars/word")
    else:
        lines.append("  (no detection TPs – metrics unavailable)")
    lines.append("")

    # Calibration (ECE)
    lines.append("--- Confidence Calibration (ECE) ---")
    lines.append(f"  ECE  Detection   : {ece_det:.6f}")
    lines.append(f"  ECE  End-to-End  : {ece_e2e:.6f}")
    lines.append("")
    lines.append("  Detection calibration bins (conf -> accuracy):")
    lines.append(f"  {'conf_lo':>7s} {'conf_hi':>7s} {'count':>6s}  "
                 f"{'avg_conf':>8s}  {'avg_acc':>7s}  {'gap':>6s}")
    for b in ece_det_bins:
        lines.append(
            f"  {b['lo']:7.3f} {b['hi']:7.3f} {b['count']:6d}  "
            f"{b['avg_conf']:8.4f}  {b['avg_acc']:7.4f}  {b['gap']:6.4f}"
        )
    lines.append("")
    lines.append("  End-to-End calibration bins:")
    lines.append(f"  {'conf_lo':>7s} {'conf_hi':>7s} {'count':>6s}  "
                 f"{'avg_conf':>8s}  {'avg_acc':>7s}  {'gap':>6s}")
    for b in ece_e2e_bins:
        lines.append(
            f"  {b['lo']:7.3f} {b['hi']:7.3f} {b['count']:6d}  "
            f"{b['avg_conf']:8.4f}  {b['avg_acc']:7.4f}  {b['gap']:6.4f}"
        )
    lines.append("")

    # Sample predictions
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
    lines.append("=" * W)

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

    analytics_path = out / "analytics.json"
    analytics_export = {
        "ap_det":  ap_det,   "ap_e2e":  ap_e2e,
        "ece_det": ece_det,  "ece_e2e": ece_e2e,
        "ece_det_bins": ece_det_bins,
        "ece_e2e_bins": ece_e2e_bins,
        "rec_metrics":  rec_m,
        "err_det":  err_det, "err_e2e":  err_e2e,
        "pr_det": {"precision": analytics.get("pr_det_p", []),
                   "recall":    analytics.get("pr_det_r", [])},
        "pr_e2e": {"precision": analytics.get("pr_e2e_p", []),
                   "recall":    analytics.get("pr_e2e_r", [])},
        "pr_det_sampled": pr_det_sampled,
        "pr_e2e_sampled": pr_e2e_sampled,
        "ngt": ngt,
    }
    with open(analytics_path, "w", encoding="utf-8") as f:
        json.dump(analytics_export, f, indent=2)

    print(report)
    print(f"\nSaved report    : {report_path}")
    print(f"Saved preds     : {json_path}")
    print(f"Saved sweep     : {sweep_path}")
    print(f"Saved analytics : {analytics_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser("ICDAR2019 SPTSv2 Evaluation")
    parser.add_argument("--resume", required=True,
                        help="Checkpoint .pth path")
    parser.add_argument("--test_imgs", default="Data/ICDAR2019/test_imgs",
                        help="Folder containing test images")
    parser.add_argument("--gt_json", required=True,
                        help="COCO-style GT JSON produced by build_annotations_json.py "
                             "or build_icdar2019_pipeline.py. "
                             "Build with --rec-string for reliable text decoding.")
    parser.add_argument("--output_dir", default="results/eval_ic19")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture — must match the checkpoint
    parser.add_argument("--lr_backbone",        type=float, default=0)
    parser.add_argument("--backbone",           default="resnet50")
    parser.add_argument("--dilation",           action="store_true")
    parser.add_argument("--position_embedding", default="sine")
    parser.add_argument("--enc_layers",         type=int,   default=6)
    parser.add_argument("--dec_layers",         type=int,   default=6)
    parser.add_argument("--window_size",        type=int,   default=5)
    parser.add_argument("--dim_feedforward",    type=int,   default=1024)
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--depths",             type=int,   default=6)
    parser.add_argument("--nheads",             type=int,   default=8)
    parser.add_argument("--num_queries",        type=int,   default=100,
                        help="Number of query slots (must match checkpoint)")
    parser.add_argument("--transformer_type",   default="vanilla",
                        choices=["vanilla", "linear"],
                        help="Transformer variant (must match checkpoint)")
    parser.add_argument("--pre_norm",           action="store_true")
    parser.add_argument("--masks",              action="store_true")

    # Vocabulary / data args
    # --chars must match the charset used by build_annotations_json.py and main.py
    # exactly: 94 printable ASCII starting from '!' (no leading space) followed
    # by 34 French accented characters (17 lowercase + 17 uppercase).
    # PAD index = len(chars) = 128; that is also the --pad_rec_index default.
    parser.add_argument("--bins",           type=int, default=1000)
    parser.add_argument("--chars", type=str,
                        default='!"#$%&\'()*+,-./0123456789:;<=>?@'
                                'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
                                'abcdefghijklmnopqrstuvwxyz{|}~'
                                '\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb'
                                '\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u00ff'
                                '\u00e6\u0153\u00e7'
                                '\u00c0\u00c2\u00c4\u00c9\u00c8\u00ca\u00cb'
                                '\u00ce\u00cf\u00d4\u00d9\u00db\u00dc\u0178'
                                '\u00c6\u0152\u00c7')
    parser.add_argument("--padding_bins",   type=int,   default=0)
    parser.add_argument("--pad_rec",        action="store_true")
    parser.add_argument("--pad_rec_index",  type=int,   default=128)
    parser.add_argument("--no_known_char",  type=int,   default=130)
    parser.add_argument("--max_length",     type=int,   default=25)
    parser.add_argument("--num_box",        type=int,   default=60)
    parser.add_argument("--obj_num",        type=int,   default=60)
    parser.add_argument("--pts_key",        default="center_pts")
    parser.add_argument("--max_size_test",  type=int,   default=1824)
    parser.add_argument("--min_size_test",  type=int,   default=1024)

    # Evaluation thresholds
    parser.add_argument("--conf_rec", type=float, default=0.62,
                        help="Per-character recognition hard threshold")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    # pad_rec must always be True (matches training config; see main.py)
    args.pad_rec = True

    args = process_args(args)

    print(f"Token indices: category_start={args.category_start_index}, "
          f"end={args.end_index}, start={args.start_index}")

    # Read GT ------------------------------------------------------------------
    print("Reading GT...")
    gt_dict = read_coco_gt(args.gt_json, decode_chars=list(args.chars))

    total_gt = sum(1 for items in gt_dict.values() for it in items if not it["dontcare"])
    total_dc = sum(1 for items in gt_dict.values() for it in items if it["dontcare"])
    print(f"GT: {len(gt_dict)} images, {total_gt} eval instances, {total_dc} dontcare")

    # Build model --------------------------------------------------------------
    print("Building model...")
    device = torch.device(args.device)
    model, _ = build_model(args)
    model.to(device)
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {args.resume}")

    # Inference ----------------------------------------------------------------
    print("Running inference...")
    transform = build_transform(args)
    preds, stats = run_inference(model, args, args.test_imgs, gt_dict, transform)
    preds.sort(key=lambda x: -x["score"])

    # Threshold sweep ----------------------------------------------------------
    print(f"Evaluating (threshold sweep, conf_rec_thr={args.conf_rec:.3f})...")
    rows, best_det, best_e2e = sweep_thresholds(
        preds, gt_dict, conf_rec_thr=args.conf_rec, lo=0.3, hi=0.95)

    # Rich analytics -----------------------------------------------------------
    print("Computing AP, ECE, recognition quality, error analysis...")
    labeled, ngt = analyze_predictions(preds, gt_dict, conf_rec_thr=args.conf_rec)

    ap_det, pr_det_p, pr_det_r = compute_ap(labeled, ngt, task="det")
    ap_e2e, pr_e2e_p, pr_e2e_r = compute_ap(labeled, ngt, task="e2e")

    ece_det, ece_det_bins = compute_ece(labeled, task="det")
    ece_e2e, ece_e2e_bins = compute_ece(labeled, task="e2e")

    rec_metrics = compute_rec_metrics(labeled)

    err_det = compute_error_analysis(labeled, best_det["threshold"], args.conf_rec)
    err_e2e = compute_error_analysis(labeled, best_e2e["threshold"], args.conf_rec)

    pr_det_sampled = _sample_pr_at_recalls(pr_det_p, pr_det_r)
    pr_e2e_sampled = _sample_pr_at_recalls(pr_e2e_p, pr_e2e_r)

    analytics = {
        "ngt":            ngt,
        "ap_det":         ap_det,
        "ap_e2e":         ap_e2e,
        "ece_det":        ece_det,
        "ece_e2e":        ece_e2e,
        "ece_det_bins":   ece_det_bins,
        "ece_e2e_bins":   ece_e2e_bins,
        "rec_metrics":    rec_metrics,
        "err_det":        err_det,
        "err_e2e":        err_e2e,
        "pr_det_p":       pr_det_p,
        "pr_det_r":       pr_det_r,
        "pr_e2e_p":       pr_e2e_p,
        "pr_e2e_r":       pr_e2e_r,
        "pr_det_sampled": pr_det_sampled,
        "pr_e2e_sampled": pr_e2e_sampled,
    }

    print(f"  AP det={ap_det:.4f}  AP e2e={ap_e2e:.4f}  "
          f"ECE det={ece_det:.4f}  ECE e2e={ece_e2e:.4f}")

    # Write report -------------------------------------------------------------
    write_report(args.output_dir, preds, stats, rows, best_det, best_e2e,
                 analytics, gt_dict, args)


if __name__ == "__main__":
    main()
