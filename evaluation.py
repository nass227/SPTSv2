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

def extract_all_gt(target, chars, img_w, img_h, pad_rec_index=96, no_known_char=95):
    """
    Extrait tous les polygones et textes GT.
    bezier_pts : (N, 16) → 8 points normalisés 0-1
    rec        : (N, 25) → indices caractères (same decoding as convert_rec_to_str)
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
                if 0 <= int(i) < len(chars)
                and int(i) != pad_rec_index
                and int(i) != no_known_char
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
                     bins=1000, padding_bins=0,
                     category_start_index=1000,
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

        # Same coordinate decode as util/visualize.py (padding_bins from main.py)
        x_bin = min(max(x_bin, 0), category_start_index - 1) - padding_bins
        y_bin = min(max(y_bin, 0), category_start_index - 1) - padding_bins
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
    output_dir, bins=1000, padding_bins=0,
    category_start_index=1000, text_length=25,
    img_size=640, max_vis=20, pad_rec_index=96, no_known_char=95,
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

                # ── taille réelle (orig_size, same as engine_sptsv2.evaluate) ─
                if "orig_size" in target and isinstance(target["orig_size"], torch.Tensor):
                    img_h_real = float(target["orig_size"][0])
                    img_w_real = float(target["orig_size"][1])
                elif "size" in target and isinstance(target["size"], torch.Tensor):
                    img_h_real = float(target["size"][0])
                    img_w_real = float(target["size"][1])
                else:
                    img_h_real = img_size
                    img_w_real = img_size

                out_i = out_tensor[i]

                # ── GT ───────────────────────────────────────────────────
                gt_polys, gt_texts = extract_all_gt(
                    target, chars, img_w_real, img_h_real,
                    pad_rec_index=pad_rec_index,
                    no_known_char=no_known_char,
                )

                # ── Prédictions ──────────────────────────────────────────
                pred_centers, pred_polys, pred_texts, pred_scores = \
                    extract_all_pred(
                        out_i, chars, img_w_real, img_h_real,
                        bins=bins,
                        padding_bins=padding_bins,
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
# Align eval args with training (main.py / checkpoint)
# ─────────────────────────────────────────────

_TRAIN_ARG_KEYS = (
    'pad_rec', 'padding_bins', 'bins', 'chars', 'max_length',
    'pad_rec_index', 'no_known_char', 'pre_norm', 'backbone',
    'enc_layers', 'dec_layers', 'window_size', 'obj_num', 'num_box',
    'dim_feedforward', 'hidden_dim', 'dropout', 'depths', 'nheads',
    'num_queries', 'transformer_type', 'dilation', 'position_embedding',
    'pts_key', 'max_size_test', 'min_size_test',
)


def merge_checkpoint_args(args, checkpoint):
    """Restore training-time token/index and model args from a checkpoint."""
    if not isinstance(checkpoint, dict) or 'args' not in checkpoint:
        return args
    ckpt_args = checkpoint['args']
    if hasattr(ckpt_args, '__dict__'):
        ckpt_args = vars(ckpt_args)
    for key in _TRAIN_ARG_KEYS:
        if key in ckpt_args:
            setattr(args, key, ckpt_args[key])
    return args


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from models import build_model
    from main import get_args_parser

    # Same CLI as main.py (incl. --max_size_test / --min_size_test, lines 116-117)
    parser = get_args_parser()
    parser.prog = "SPTSv2-ResNet evaluation"
    parser.set_defaults(output_dir="./evaluation_results")

    # ── Eval-only (do not re-add flags already in get_args_parser) ─────────
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_vis", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=None,
                        help="Fallback image size when target has no orig_size "
                             "(defaults to --min_size_test)")

    args = parser.parse_args()

    if not args.val_dataset or not args.data_root:
        parser.error("--val_dataset and --data_root are required for evaluation")

    print(f"Chargement : {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)

    # Match padding / indexes / model config to the training run when possible
    merge_checkpoint_args(args, checkpoint)

    # Same index computation as main.py (process_args in util/data.py)
    args = process_args(args)

    if args.img_size is None:
        args.img_size = args.min_size_test

    print(f"Test resize: max_size_test={args.max_size_test}, "
          f"min_size_test={args.min_size_test}")
    print(f"Token indices: category_start={args.category_start_index}, "
          f"end={args.end_index}, start={args.start_index}, "
          f"padding_bins={args.padding_bins}, pad_rec={args.pad_rec}, "
          f"pad_rec_index={args.pad_rec_index}, no_known_char={args.no_known_char}")

    # ── Build model ─────────────────────────────────────────────────────
    model, _ = build_model(args)

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
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = evaluate_complete(
        model                = model,
        data_loader          = data_loader,
        device               = args.device,
        chars                = args.chars,
        start_index          = args.start_index,
        output_dir           = args.output_dir,
        bins                 = args.bins,
        padding_bins         = args.padding_bins,
        category_start_index = args.category_start_index,
        text_length          = args.max_length,
        img_size             = args.img_size,
        max_vis              = args.max_vis,
        pad_rec_index        = args.pad_rec_index,
        no_known_char        = args.no_known_char,
    )

    save_results(results, args.output_dir)


