# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
from typing import Iterable
from tqdm import tqdm
from torch.cuda.amp import autocast

import util.misc_sptsv2 as utils
from util.visualize import vis_output_seqs, convert_rec_to_str


def _accumulate_gt(gt_by_image, target, chars):
    """Collect GT centers and recognition strings (same decoding as training labels)."""
    image_id = int(target["image_id"].item())
    oh, ow = int(target["orig_size"][0].item()), int(target["orig_size"][1].item())
    centers = target["center_pts"].detach().cpu().numpy()
    rec = target["rec"].detach().cpu()
    if image_id not in gt_by_image:
        gt_by_image[image_id] = {"orig_h": oh, "orig_w": ow, "items": []}
    for i in range(centers.shape[0]):
        text = convert_rec_to_str([rec[i]], chars)[0]
        gt_by_image[image_id]["items"].append(
            {"cx": float(centers[i, 0]), "cy": float(centers[i, 1]), "text": text}
        )


def _match_dist_threshold(orig_h, orig_w):
    """Loose center match for quick det check (pixels)."""
    return max(24.0, 0.015 * math.hypot(float(orig_h), float(orig_w)))


def _greedy_match_centers(gt_items, preds, oh, ow):
    """
    One-to-one greedy match: each GT paired with closest unused pred within threshold.
    Returns list of dicts with pred index, gt index, dist, rec_match (bool).
    """
    thr = _match_dist_threshold(oh, ow)
    used = set()
    matches = []
    for gi, g in enumerate(gt_items):
        best_j = None
        best_d = thr + 1.0
        for j, p in enumerate(preds):
            if j in used:
                continue
            px, py = p["polys"][0][0], p["polys"][0][1]
            d = math.hypot(px - g["cx"], py - g["cy"])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= thr:
            used.add(best_j)
            pr = preds[best_j].get("rec", "") or ""
            gt = g.get("text", "") or ""
            matches.append(
                {
                    "gt_idx": gi,
                    "pred_idx": best_j,
                    "dist": best_d,
                    "rec_exact": pr == gt,
                    "pred_rec": pr,
                    "gt_rec": gt,
                }
            )
    return matches, used, thr

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler: list = [0], print_freq: int = 10,
                    text_length: int = 25, scaler=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    optimizer.param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer.param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1

    for samples, input_box_seqs, input_label_seqs, output_box_seqs, output_label_seqs in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        input_box_seqs = input_box_seqs.to(device)
        input_label_seqs = input_label_seqs.to(device)
        output_box_seqs = output_box_seqs.to(device)
        output_label_seqs = output_label_seqs.to(device)
        # samples = samples.to(device)

        # Debug: find the exact bad batch
        if torch.all(samples.mask):
            print(f"WARNING: fully masked batch at iteration, skipping")
            continue

        # Also check for any all-masked sample in the batch
        if samples.mask.flatten(1).all(dim=1).any():
            print(f"WARNING: batch contains a fully masked sample, skipping")
            continue



        if not all(input_label_seqs.tolist()):
            continue

        output_seqs = torch.cat([output_box_seqs.flatten(), output_label_seqs.flatten()])

        with autocast(enabled=(scaler is not None)):
            outputs_box, outputs_label = model(samples, input_box_seqs, input_label_seqs, text_length)
            outputs_box   = outputs_box.reshape(-1, outputs_box.shape[-1])
            outputs_label = outputs_label.reshape(-1, outputs_label.shape[-1])
            outputs = torch.cat([outputs_box, outputs_label], 0)
            loss    = criterion(outputs, output_seqs.flatten())

        loss_dict  = {'at': loss}
        weight_dict = {'at': 1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        loss_dict_reduced         = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled   = {k: v * weight_dict[k]
                                      for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     print(loss_dict_reduced)
        #     sys.exit(1)

        optimizer.zero_grad()  # ← move this UP, before the isfinite check

        if not math.isfinite(loss_value):
            print(f"WARNING: Non-finite loss {loss_value}, skipping batch")
            torch.cuda.empty_cache() 
            continue

        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # optimizer.zero_grad()
        # if scaler is not None:
        #     scaler.scale(losses).backward()
        #     if max_norm > 0:
        #         scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     losses.backward()
        #     if max_norm > 0:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        #     optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_loss(model: torch.nn.Module, criterion: torch.nn.Module,
                  data_loader: Iterable, device: torch.device,
                  epoch: int, text_length: int = 25):
    """
    Compute teacher-forcing cross-entropy loss on the validation split,
    without any gradient computation or parameter updates.

    The full model is put in ``train()`` for these forwards (same as
    ``train_one_epoch``): both ``SPTSv2.forward`` and ``Transformer.forward``
    branch on ``self.training``. In eval mode the transformer runs autoregressive
    inference (CUDA gather issues) while ``SPTSv2`` still expects inference-shaped
    tensors, causing shape errors. Teacher-forcing loss requires the training
    path on the whole module tree. Prior train/eval state is restored on exit.
    """
    prev_training = model.training
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Val Epoch: [{epoch}]'

    try:
        for samples, input_box_seqs, input_label_seqs, output_box_seqs, output_label_seqs in \
                metric_logger.log_every(data_loader, 10, header):

            samples           = samples.to(device)
            input_box_seqs    = input_box_seqs.to(device)
            input_label_seqs  = input_label_seqs.to(device)
            output_box_seqs   = output_box_seqs.to(device)
            output_label_seqs = output_label_seqs.to(device)

            if not all(input_label_seqs.tolist()):
                continue

            output_seqs = torch.cat([output_box_seqs.flatten(), output_label_seqs.flatten()])

            outputs_box, outputs_label = model(samples, input_box_seqs, input_label_seqs, text_length)
            outputs_box   = outputs_box.reshape(-1, outputs_box.shape[-1])
            outputs_label = outputs_label.reshape(-1, outputs_label.shape[-1])
            outputs       = torch.cat([outputs_box, outputs_label], 0)
            loss          = criterion(outputs, output_seqs.flatten())

            loss_dict   = {'at': loss}
            weight_dict = {'at': 1}

            loss_dict_reduced          = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled   = {k: v * weight_dict[k]
                                          for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                continue

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

    finally:
        model.train(prev_training)

    metric_logger.synchronize_between_processes()
    print("Val stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def decode_recognition_directly(output_seq, chars, category_start_index=1000):
    """
    Decode recognition directly from model output sequence using the same logic as predict.py
    """
    text = ''
    for c in output_seq:
        c_val = c.item() if torch.is_tensor(c) else c
        if category_start_index <= c_val < category_start_index + len(chars):
            text += chars[c_val - category_start_index]
        else:
            break
    return text

@torch.no_grad()
def evaluate(
    model,
    criterion,
    data_loader,
    device,
    output_dir,
    chars,
    start_index,
    category_start_index,
    visualize=False,
    text_length=25,
):
    model.eval()
    criterion.eval()
    chars = list(chars)
    import time
    cnt = 0
    total = 0
    results = []
    total_candidate_predictions = 0
    empty_rec_predictions = 0
    printed_candidates = 0
    gt_by_image = {}
    eval_dataset_name = None
    dataset_names = []

    for samples, targets in tqdm(data_loader):
        batch = len(targets)
        targets = targets[: batch // 2]
        for t in targets:
            _accumulate_gt(gt_by_image, t, chars)
        if eval_dataset_name is None and targets:
            dname = targets[0]["dataset_name"]
            eval_dataset_name = dname.item() if torch.is_tensor(dname) else str(dname)
        samples.mask = samples.mask[: batch // 2, :, :]
        samples.tensors = samples.tensors[: batch // 2, :, :, :]
        samples = samples.to(device)
        dataset_names = [target['dataset_name'] for target in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'dataset_name'} for t in targets]
        seq = torch.ones(len(targets), 1).to(samples.mask) * start_index
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(samples, seq, seq, text_length)
        torch.cuda.synchronize()
        t1 = time.time()
        cnt += 1
        total += t1-t0
        print(f"Average inference time: {total/cnt:.4f}s")
        
        if outputs == None:
            continue
        outputs, values, rec_scores = outputs
        
        if visualize:
            samples_ = samples.to(torch.device('cpu')); outputs_ = outputs.cpu()
            vis_images = vis_output_seqs(samples_, outputs_, rec_scores, False, True, text_length, chars)
            for vis_image, target, dataset_name in zip(vis_images, targets, dataset_names):
                save_path = os.path.join(output_dir, 'vis', dataset_name, '{:06d}.jpg'.format(target['image_id'].item()))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, vis_image)

        outputs = outputs.cpu(); values = values.cpu(); rec_scores = rec_scores.cpu()

        for target, output, value, rec_score in zip(targets, outputs, values, rec_scores):
            image_id = target["image_id"].item()
            orig_h, orig_w = target["orig_size"]
            pred_length = text_length + 2
            n_pred = output.shape[0] // pred_length

            for i in range(n_pred):
                s = i * pred_length
                x_abs = output[s].item() * (orig_w.item() / 1000.0)
                y_abs = output[s + 1].item() * (orig_h.item() / 1000.0)
                token_seq = output[s + 2:s + pred_length]
                rec_label = decode_recognition_directly(
                    token_seq, chars, category_start_index=category_start_index
                )
                score = float(value[s:s + pred_length].mean().item())
                is_empty = (len(rec_label) == 0)

                if printed_candidates < 40:
                    token_preview = token_seq[:8].tolist()
                    print(
                        f"[eval cand {printed_candidates:03d}] img={image_id} "
                        f"conf={score:.4f} xy=({x_abs:.1f},{y_abs:.1f}) "
                        f"empty={is_empty} text='{rec_label}' tokens={token_preview}"
                    )
                    printed_candidates += 1

                if rec_label:
                    rec_score_slice = rec_score[s + 2:s + pred_length]
                    result = {
                        "image_id": image_id,
                        "category_id": 1,
                        "polys": [[x_abs, y_abs]],
                        "rec": rec_label,
                        "score": score,
                        "rec_score": rec_score_slice.numpy().tolist(),
                    }
                    results.append(result)
                else:
                    empty_rec_predictions += 1
                total_candidate_predictions += 1

    avg_infer_s = (total / cnt) if cnt else 0.0
    dataset_name = eval_dataset_name or (dataset_names[0] if dataset_names else "unknown")

    # Save results JSON
    if len(results) > 0:
        json_path = os.path.join(output_dir, 'results', str(dataset_name) + '.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        results_json = json.dumps(results, indent=4)
        with open(json_path, 'w') as f:
            f.write(results_json)
        print(f"Saved {len(results)} predictions to {json_path}")
    else:
        print("WARNING: No predictions with non-empty text found!")

    compute_quick_metrics(
        results,
        gt_by_image,
        output_dir,
        dataset_name,
        avg_infer_s,
        visualize,
        total_candidate_predictions,
        empty_rec_predictions,
    )


def compute_quick_metrics(
    results,
    gt_by_image,
    output_dir,
    dataset_name,
    avg_infer_s,
    visualize,
    total_candidate_predictions,
    empty_rec_predictions,
):
    """
    Quick det/rec sanity metrics vs GT centers + labels, and a text report with samples.
    """
    pred_by_image = {}
    for pred in results:
        img_id = pred["image_id"]
        pred_by_image.setdefault(img_id, []).append(pred)

    total_gt = sum(len(v["items"]) for v in gt_by_image.values())
    total_pred = len(results)
    matched = 0
    rec_exact_on_match = 0
    dist_sum = 0.0
    all_matches = []
    thr_ref = None

    for image_id, ginfo in gt_by_image.items():
        oh, ow = ginfo["orig_h"], ginfo["orig_w"]
        thr_ref = _match_dist_threshold(oh, ow)
        preds = pred_by_image.get(image_id, [])
        gitems = ginfo["items"]
        matches, used, thr = _greedy_match_centers(gitems, preds, oh, ow)
        matched += len(matches)
        for m in matches:
            if m["rec_exact"]:
                rec_exact_on_match += 1
            dist_sum += m["dist"]
            m["image_id"] = image_id
            all_matches.append(m)
        # unused preds on this image (for samples)
        for j, p in enumerate(preds):
            if j not in used:
                all_matches.append(
                    {
                        "image_id": image_id,
                        "pred_idx": j,
                        "unmatched_pred": True,
                        "pred_rec": p.get("rec", ""),
                        "score": p.get("score", 0.0),
                        "px": p["polys"][0][0],
                        "py": p["polys"][0][1],
                    }
                )

    det_recall = matched / total_gt if total_gt else 0.0
    det_precision = matched / total_pred if total_pred else 0.0
    det_f1 = (
        (2 * det_precision * det_recall / (det_precision + det_recall))
        if (det_precision + det_recall) > 0
        else 0.0
    )
    rec_acc = rec_exact_on_match / matched if matched else 0.0
    mean_dist = dist_sum / matched if matched else 0.0

    lines = []
    lines.append("SPTS quick evaluation report")
    lines.append("dataset: {}".format(dataset_name))
    lines.append("avg_batch_inference_s: {:.4f}".format(avg_infer_s))
    lines.append("center_match_threshold_px (typical): {:.2f}".format(thr_ref if thr_ref is not None else 0.0))
    lines.append("---")
    lines.append("counts: gt_instances={} pred_instances_non_empty={}".format(total_gt, total_pred))
    lines.append(
        "recognition empty strings: {} / {} ({:.2f}%)".format(
            empty_rec_predictions,
            total_candidate_predictions,
            (100.0 * empty_rec_predictions / total_candidate_predictions) if total_candidate_predictions else 0.0,
        )
    )
    lines.append("detection (greedy center match): matched={}".format(matched))
    lines.append("  recall {:.4f}  precision {:.4f}  f1 {:.4f}".format(det_recall, det_precision, det_f1))
    lines.append("recognition (on matched pairs): exact_acc {:.4f}  ({}/{})".format(
        rec_acc, rec_exact_on_match, matched))
    lines.append("localization: mean_L2_px_on_matches {:.2f}".format(mean_dist))
    lines.append("---")
    lines.append("sample predictions (raw, first up to 12):")
    for pred in results[:12]:
        lines.append(
            "  img {} | '{}' | score {:.3f} | ({:.1f}, {:.1f})".format(
                pred["image_id"],
                pred.get("rec", ""),
                pred.get("score", 0.0),
                pred["polys"][0][0],
                pred["polys"][0][1],
            )
        )
    lines.append("---")
    lines.append("matched pairs (pred vs GT), up to 20:")
    shown = 0
    for m in all_matches:
        if shown >= 20:
            break
        if m.get("unmatched_pred"):
            lines.append(
                "  img {} | UNMATCHED pred | '{}' | score {:.3f} | ({:.1f}, {:.1f})".format(
                    m["image_id"],
                    m["pred_rec"],
                    m.get("score", 0.0),
                    m["px"],
                    m["py"],
                )
            )
            shown += 1
            continue
        lines.append(
            "  img {} | pred: '{}' | gt: '{}' | dist {:.1f}px | rec_ok {}".format(
                m["image_id"],
                m["pred_rec"],
                m["gt_rec"],
                m["dist"],
                m["rec_exact"],
            )
        )
        shown += 1
    if visualize:
        vis_dir = os.path.join(output_dir, "vis", str(dataset_name))
        lines.append("---")
        lines.append("visualization samples saved under: {}".format(vis_dir))

    report = "\n".join(lines) + "\n"
    out_dir = os.path.join(output_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "quick_eval_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print(report, end="")
    print("Wrote {}".format(report_path))
    print("=" * 50)