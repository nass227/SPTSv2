import os
os.environ['MPLBACKEND'] = 'Agg'
import json
import torch
import numpy as np
from tqdm import tqdm
import editdistance as ed
from shapely.geometry import Polygon, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from util.data import process_args
import util.misc_sptsv2 as utils
from datasets import build_dataset


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────

def compute_center_hit(pred_cx, pred_cy, gt_poly):
    """
    Retourne 1 si le centre prédit est à l'intérieur du polygone GT.
    C'est la métrique correcte car le modèle prédit un centre, pas une box.
    """
    try:
        poly = Polygon(gt_poly)
        pt   = Point(pred_cx, pred_cy)
        return 1 if poly.contains(pt) else 0
    except Exception:
        return 0


def compute_center_distance(pred_cx, pred_cy, gt_poly):
    """Distance entre centre prédit et centre du GT."""
    try:
        gt_cx = np.mean([p[0] for p in gt_poly])
        gt_cy = np.mean([p[1] for p in gt_poly])
        return float(np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2))
    except Exception:
        return float('inf')


def compute_cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0
    return ed.eval(reference, hypothesis) / len(reference)


def compute_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0
    return ed.eval(ref_words, hyp_words) / len(ref_words)


def compute_word_accuracy(predictions, ground_truths):
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)


def compute_map_center(pred_centers, gt_polys):
    """
    mAP basé sur si le centre prédit est dans le polygone GT.
    Plus adapté que l'IoU pour un modèle qui prédit des centres.
    """
    if not gt_polys or not pred_centers:
        return 0.0

    tp = np.zeros(len(pred_centers))
    fp = np.zeros(len(pred_centers))
    matched = set()

    for i, (cx, cy, score) in enumerate(pred_centers):
        best_hit, best_j = 0, -1
        for j, gt_poly in enumerate(gt_polys):
            if j in matched:
                continue
            hit = compute_center_hit(cx, cy, gt_poly)
            if hit > best_hit:
                best_hit, best_j = hit, j
        if best_hit == 1:
            tp[i] = 1
            matched.add(best_j)
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls    = tp_cum / (len(gt_polys) + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls    = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


# ─────────────────────────────────────────────
# Extraction GT
# ─────────────────────────────────────────────

def extract_all_gt(target, chars, img_w, img_h):
    """
    Extrait tous les polygones et textes GT.
    bezier_pts : (N, 16) → 8 points normalisés 0-1
    rec        : (N, 25) → indices caractères, 128=padding
    """
    polygons = []
    texts    = []

    if "bezier_pts" not in target:
        return polygons, texts

    pts = target["bezier_pts"]
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    pts = np.array(pts)

    rec = target.get("rec", None)
    if isinstance(rec, torch.Tensor):
        rec = rec.cpu().numpy()

    for k in range(len(pts)):
        pts_k   = pts[k].reshape(8, 2)
        corners = [pts_k[0], pts_k[3], pts_k[4], pts_k[7]]
        poly    = [[float(p[0]) * img_w, float(p[1]) * img_h]
                   for p in corners]
        polygons.append(poly)

        if rec is not None and k < len(rec):
            indices = rec[k]
            txt = "".join(
                chars[int(i)] for i in indices
                if 0 <= int(i) < len(chars) and int(i) != 128
            )
            texts.append(txt.strip())
        else:
            texts.append("")

    return polygons, texts


# ─────────────────────────────────────────────
# Extraction prédictions
# ─────────────────────────────────────────────

def estimate_box_size(gt_polys):
    """Taille moyenne des mots GT → utilisée pour visualisation seulement."""
    widths, heights = [], []
    for poly in gt_polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        widths.append(max(xs) - min(xs))
        heights.append(max(ys) - min(ys))
    if widths:
        return np.mean(widths) / 2, np.mean(heights) / 2
    return 60, 25


def extract_all_pred(out_tensor, chars, img_w, img_h,
                     bins=1000, category_start_index=1000,
                     text_length=25, gt_polys=None):
    """
    Decode predictions from the ResNet SPTSv2 output tensor.

    Format — fixed-stride blocks of pred_length = text_length + 2 tokens:
      [x_bin, y_bin, char_tok_0, char_tok_1, ..., char_tok_{text_length-1}]

      x_bin, y_bin        : coordinate bins  (0 – bins-1)
      char_tok_k          : category_start_index + char_idx
                            decoding stops at first out-of-range token
    """
    pred_length = text_length + 2
    centers  = []
    polygons = []
    texts    = []
    scores   = []

    if gt_polys and len(gt_polys) > 0:
        hw, hh = estimate_box_size(gt_polys)
    else:
        hw, hh = 60, 25

    seq    = out_tensor.tolist()
    n_pred = len(seq) // pred_length

    for i in range(n_pred):
        s     = i * pred_length
        x_bin = seq[s]
        y_bin = seq[s + 1]

        cx = x_bin / bins * img_w
        cy = y_bin / bins * img_h

        # Decode characters — stop at first non-character token (pad / EOS)
        word = []
        for tok in seq[s + 2: s + pred_length]:
            if category_start_index <= tok < category_start_index + len(chars):
                word.append(chars[tok - category_start_index])
            else:
                break

        text = "".join(word).strip()
        if not text:
            continue

        poly = [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ]

        centers.append((cx, cy, 1.0))
        polygons.append(poly)
        texts.append(text)
        scores.append(1.0)

    return centers, polygons, texts, scores


# ─────────────────────────────────────────────
# Image utils
# ─────────────────────────────────────────────

def tensor_to_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def draw_predictions_on_image(
    image_np, pred_centers, pred_polygons, pred_texts,
    gt_polygons=None, gt_texts=None,
):
    img_pil = Image.fromarray(image_np)
    draw    = ImageDraw.Draw(img_pil)

    try:
        font       = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_small = ImageFont.load_default()

    # ── GT en vert ──────────────────────────────────────────────────────
    if gt_polygons:
        for poly, txt in zip(gt_polygons, gt_texts or []):
            if poly is None:
                continue
            pts_flat = [coord for pt in poly for coord in pt]
            draw.polygon(pts_flat, outline=(0, 200, 0))
            if txt:
                x0 = min(p[0] for p in poly)
                y0 = max(min(p[1] for p in poly) - 14, 0)
                draw.text((x0, y0), f"GT: {txt}",
                          fill=(0, 200, 0), font=font_small)

    # ── Prédictions en rouge ─────────────────────────────────────────────
    for (cx, cy, _), poly, txt in zip(pred_centers, pred_polygons, pred_texts):

        # box estimée (pour visualisation)
        pts_flat = [coord for pt in poly for coord in pt]
        draw.polygon(pts_flat, outline=(255, 60, 60))

        # point central prédit (cercle rouge)
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(255, 0, 0), outline=(255, 0, 0))

        # texte prédit
        label = txt if txt else "?"
        try:
            bbox_text = draw.textbbox((cx, max(cy - 20, 0)), label, font=font)
            draw.rectangle(bbox_text, fill=(0, 0, 0))
        except Exception:
            pass
        draw.text((cx, max(cy - 20, 0)), label, fill=(255, 60, 60), font=font)

    return np.array(img_pil)


def save_visualization(image_np, pred_centers, pred_polygons, pred_texts,
                       gt_polygons, gt_texts, save_path, image_name=""):
    annotated = draw_predictions_on_image(
        image_np, pred_centers, pred_polygons, pred_texts,
        gt_polygons, gt_texts,
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(image_np)
    axes[0].set_title("Image originale", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(annotated)
    axes[1].set_title(
        f"Prédictions SPTSv2-ResNet — {image_name}\n"
        f"Rouge=Centre prédit  Vert=GT box", fontsize=13)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Évaluation principale
# ─────────────────────────────────────────────

def evaluate_complete(
    model, data_loader, device, chars, start_index,
    output_dir, bins=1000, category_start_index=1000,
    text_length=25, img_size=640, max_vis=20,
):
    model.eval()
    chars = list(chars)

    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    all_hits, all_distances           = [], []
    all_precisions, all_recalls       = [], []
    all_f1s                           = []
    all_cer, all_wer                  = [], []
    all_predictions, all_gts          = [], []
    all_map_aps                       = []
    vis_count = 0

    print("Évaluation en cours...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            if len(batch) != 2:
                continue

            samples, targets = batch
            samples = samples.to(device)

            seq = torch.ones(
                len(targets), 1, dtype=torch.long, device=device
            ) * start_index

            try:
                outputs = model(samples, seq, seq, text_length)
            except Exception as e:
                print(f"Erreur inférence batch {batch_idx} : {e}")
                continue

            if outputs is None:
                continue

            out_tensor, _, rec_scores = outputs
            out_tensor = out_tensor.cpu()
            rec_scores = rec_scores.cpu()

            for i, target in enumerate(targets):

                # ── taille réelle ────────────────────────────────────────
                if "size" in target and isinstance(target["size"], torch.Tensor):
                    img_h_real = float(target["size"][0])
                    img_w_real = float(target["size"][1])
                else:
                    img_h_real = img_size
                    img_w_real = img_size

                out_i = out_tensor[i]

                # ── GT ───────────────────────────────────────────────────
                gt_polys, gt_texts = extract_all_gt(
                    target, chars, img_w_real, img_h_real
                )

                # ── Prédictions ──────────────────────────────────────────
                pred_centers, pred_polys, pred_texts, pred_scores = \
                    extract_all_pred(
                        out_i, chars, img_w_real, img_h_real,
                        bins=bins,
                        category_start_index=category_start_index,
                        text_length=text_length,
                        gt_polys=gt_polys,
                    )

                # ── Métriques localisation ───────────────────────────────
                # Pour chaque centre prédit, chercher le meilleur GT
                matched_gt   = set()
                matched_pred = set()

                for pred_idx, (cx, cy, _) in enumerate(pred_centers):
                    best_hit, best_gt_idx = 0, -1
                    for gt_idx, gt_poly in enumerate(gt_polys):
                        if gt_idx in matched_gt:
                            continue
                        hit = compute_center_hit(cx, cy, gt_poly)
                        if hit > best_hit:
                            best_hit, best_gt_idx = hit, gt_idx

                    all_hits.append(best_hit)

                    if best_hit == 1:
                        matched_pred.add(pred_idx)
                        matched_gt.add(best_gt_idx)
                    else:
                        # distance au GT le plus proche
                        if gt_polys:
                            dists = [compute_center_distance(cx, cy, g)
                                     for g in gt_polys]
                            all_distances.append(min(dists))

                tp = len(matched_pred)
                fp = len(pred_centers) - tp
                fn = len(gt_polys)     - tp

                precision = tp / (tp + fp + 1e-6)
                recall    = tp / (tp + fn + 1e-6)
                f1        = 2 * precision * recall / (precision + recall + 1e-6)

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)

                # mAP basé sur center-hit
                ap = compute_map_center(pred_centers, gt_polys)
                all_map_aps.append(ap)

                # ── Métriques reconnaissance ─────────────────────────────
                pred_text_full = " ".join(pred_texts)
                gt_text_full   = " ".join(gt_texts)

                if gt_text_full:
                    all_cer.append(compute_cer(
                        gt_text_full.lower(), pred_text_full.lower()))
                    all_wer.append(compute_wer(
                        gt_text_full.lower(), pred_text_full.lower()))

                all_predictions.append(pred_text_full)
                all_gts.append(gt_text_full)

                # ── Visualisation ────────────────────────────────────────
                if vis_count < max_vis:
                    try:
                        img_np    = tensor_to_image(samples.tensors[i])
                        save_path = os.path.join(
                            vis_dir, f"sample_{batch_idx:04d}_{i}.png"
                        )
                        save_visualization(
                            image_np      = img_np,
                            pred_centers  = pred_centers,
                            pred_polygons = pred_polys,
                            pred_texts    = pred_texts,
                            gt_polygons   = gt_polys,
                            gt_texts      = gt_texts,
                            save_path     = save_path,
                            image_name    = f"batch{batch_idx}_img{i}",
                        )
                        vis_count += 1
                    except Exception as e:
                        print(f"Erreur visualisation : {e}")

    results = {
        "Centre dans GT (%)": float(np.mean(all_hits)) * 100
                               if all_hits else 0.0,
        "Distance moyenne"  : float(np.mean(all_distances))
                               if all_distances else 0.0,
        "Précision"         : float(np.mean(all_precisions))
                               if all_precisions else 0.0,
        "Rappel"            : float(np.mean(all_recalls))
                               if all_recalls else 0.0,
        "F1-score"          : float(np.mean(all_f1s))
                               if all_f1s else 0.0,
        "mAP (center-hit)"  : float(np.mean(all_map_aps))
                               if all_map_aps else 0.0,
        "CER"               : float(np.mean(all_cer))
                               if all_cer else 0.0,
        "WER"               : float(np.mean(all_wer))
                               if all_wer else 0.0,
        "Word Accuracy"     : compute_word_accuracy(all_predictions, all_gts),
    }

    print(f"\n{vis_count} images sauvegardées dans : {vis_dir}")
    return results


# ─────────────────────────────────────────────
# Sauvegarde & graphique
# ─────────────────────────────────────────────

def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("RÉSULTATS DE L'ÉVALUATION — SPTSv2-ResNet")
    print("=" * 55)
    for metric, value in results.items():
        if metric == "Distance moyenne":
            print(f"{metric:25s}: {value:.1f} px")
        elif metric == "Centre dans GT (%)":
            print(f"{metric:25s}: {value:.2f}%")
        elif metric in ("CER", "WER"):
            print(f"{metric:25s}: {value:.4f}")
        else:
            print(f"{metric:25s}: {value:.2%}")
    print("=" * 55)
    print(f"Résultats sauvegardés dans : {json_path}")
    create_visualization(results, output_dir)


def create_visualization(results, output_dir):
    # séparer métriques affichables en %
    plot_metrics = {
        k: v for k, v in results.items()
        if k != "Distance moyenne"
    }
    metrics     = list(plot_metrics.keys())
    values_norm = []
    for k, v in plot_metrics.items():
        if k == "Centre dans GT (%)":
            values_norm.append(v)
        elif k in ("CER", "WER"):
            values_norm.append(v * 100)
        else:
            values_norm.append(v * 100)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(metrics, values_norm, color="steelblue")
    plt.title("Métriques d'évaluation — SPTSv2-ResNet")
    plt.ylabel("Score (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(values_norm) * 1.15 if values_norm else 1)

    for bar, val in zip(bars, values_norm):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "evaluation_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Graphique sauvegardé dans : {chart_path}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from models import build_model

    parser = argparse.ArgumentParser("SPTSv2-ResNet evaluation")

    # ── Paths ───────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--output_dir",      type=str, default="./evaluation_results")
    parser.add_argument("--val_dataset",     type=str, required=True)
    parser.add_argument("--data_root",       type=str, required=True)
    parser.add_argument("--dataset_file",    type=str, default="ocr")

    # ── Runtime ─────────────────────────────────────────────────────────
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vis",         type=int,   default=20)
    parser.add_argument("--img_size",        type=int,   default=640,
                        help="Fallback image size when target has no 'size' field")

    # ── ResNet backbone ─────────────────────────────────────────────────
    parser.add_argument("--backbone",        type=str,   default="resnet50",
                        help="Backbone name (must match checkpoint)")
    parser.add_argument("--dilation",        action="store_true",
                        help="Replace last ResNet stride with dilation (DC5)")
    parser.add_argument("--position_embedding", type=str, default="sine",
                        choices=["sine", "learned"])
    parser.add_argument("--lr_backbone",     type=float, default=0)

    # ── Transformer ─────────────────────────────────────────────────────
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--nheads",          type=int,   default=8)
    parser.add_argument("--window_size",     type=int,   default=5)
    parser.add_argument("--enc_layers",      type=int,   default=6)
    parser.add_argument("--dec_layers",      type=int,   default=6)
    parser.add_argument("--obj_num",         type=int,   default=60)
    parser.add_argument("--dim_feedforward", type=int,   default=1024)
    parser.add_argument("--depths",          type=int,   default=6)
    parser.add_argument("--num_queries",     type=int,   default=100)
    parser.add_argument("--pre_norm",        action="store_true")
    parser.add_argument("--transformer_type",type=str,   default="vanilla",
                        choices=["vanilla", "linear"])
    parser.add_argument("--masks",           action="store_true")

    # ── Vocabulary ──────────────────────────────────────────────────────
    parser.add_argument("--bins",            type=int,   default=1000)
    parser.add_argument("--chars",           type=str,
                        default='!"#$%&\'()*+,-./0123456789:;<=>?@'
                                'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
                                'abcdefghijklmnopqrstuvwxyz{|}~'
                                '\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb'
                                '\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u00ff'
                                '\u00e6\u0153\u00e7'
                                '\u00c0\u00c2\u00c4\u00c9\u00c8\u00ca\u00cb'
                                '\u00ce\u00cf\u00d4\u00d9\u00db\u00dc\u0178'
                                '\u00c6\u0152\u00c7')
    parser.add_argument("--padding_bins",    type=int,   default=0)
    parser.add_argument("--num_box",         type=int,   default=60)
    parser.add_argument("--pts_key",         type=str,   default="center_pts")
    parser.add_argument("--no_known_char",   type=int,   default=130)
    parser.add_argument("--pad_rec_index",   type=int,   default=128)
    parser.add_argument("--pad_rec",         action="store_true")
    parser.add_argument("--dict_name",       type=str,   default="en_US.dic")
    parser.add_argument("--use_dict",        action="store_true")
    parser.add_argument("--max_length",      type=int,   default=25)

    # ── Data augmentation (needed by build_dataset) ──────────────────────
    parser.add_argument("--max_size_train",  type=int,   default=1600)
    parser.add_argument("--min_size_train",  type=int,   nargs="+",
                        default=[640, 672, 704, 736, 768, 800, 832, 864, 896])
    parser.add_argument("--max_size_test",   type=int,   default=1824)
    parser.add_argument("--min_size_test",   type=int,   default=1024)
    parser.add_argument("--crop_min_ratio",  type=float, default=0.5)
    parser.add_argument("--crop_max_ratio",  type=float, default=1.0)
    parser.add_argument("--crop_prob",       type=float, default=1.0)
    parser.add_argument("--rotate_max_angle",type=int,   default=30)
    parser.add_argument("--rotate_prob",     type=float, default=0.3)
    parser.add_argument("--brightness",      type=float, default=0.5)
    parser.add_argument("--contrast",        type=float, default=0.5)
    parser.add_argument("--saturation",      type=float, default=0.5)
    parser.add_argument("--hue",             type=float, default=0.5)
    parser.add_argument("--distortion_prob", type=float, default=0.5)
    parser.add_argument("--remove_difficult",action="store_true")

    # ── Mode flags (required by build_dataset / collate_fn) ─────────────
    parser.add_argument("--train",           action="store_true")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--finetune",        action="store_true")
    parser.add_argument("--visualize",       action="store_true")

    args = parser.parse_args()

    # pad_rec must be True — matches training config (see main.py)
    args.pad_rec = True
    args = process_args(args)

    print(f"Token indices: category_start={args.category_start_index}, "
          f"end={args.end_index}, start={args.start_index}")

    # ── Build model ─────────────────────────────────────────────────────
    model, _ = build_model(args)

    print(f"Chargement : {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"Checkpoint complet — epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("State dict direct chargé")

    model.to(args.device)
    model.eval()

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset_val = build_dataset(image_set="val", args=args)
    data_loader = DataLoader(
        dataset_val,
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = utils.collate_fn(args),
        num_workers = 0,
    )

    # ── Évaluation ──────────────────────────────────────────────────────
    results = evaluate_complete(
        model                = model,
        data_loader          = data_loader,
        device               = args.device,
        chars                = args.chars,
        start_index          = args.start_index,
        output_dir           = args.output_dir,
        bins                 = args.bins,
        category_start_index = args.category_start_index,
        text_length          = args.max_length,
        img_size             = args.img_size,
        max_vis              = args.max_vis,
    )

    save_results(results, args.output_dir)


