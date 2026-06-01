import os
os.environ['MPLBACKEND'] = 'Agg'
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import editdistance as ed
from shapely.geometry import Polygon, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import timm
import torchvision.models as tv_models

from util.data import process_args
import util.misc_sptsv2 as utils
from datasets import build_dataset


# ─────────────────────────────────────────────
# Métriques  (identiques à evaluation.py)
# ─────────────────────────────────────────────

def compute_center_hit(pred_cx, pred_cy, gt_poly):
    try:
        poly = Polygon(gt_poly)
        pt   = Point(pred_cx, pred_cy)
        return 1 if poly.contains(pt) else 0
    except Exception:
        return 0


def compute_center_distance(pred_cx, pred_cy, gt_poly):
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

        x_bin = min(max(x_bin, 0), category_start_index - 1) - padding_bins
        y_bin = min(max(y_bin, 0), category_start_index - 1) - padding_bins
        cx = x_bin / bins * img_w
        cy = y_bin / bins * img_h

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

def tensor_to_image(tensor, mask=None):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)

    if mask is not None:
        valid = (~mask.cpu().numpy())
        if valid.any():
            rows = np.where(valid.any(axis=1))[0]
            cols = np.where(valid.any(axis=0))[0]
            img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return img


def get_eval_image_size(target, fallback=640):
    if "size" in target and isinstance(target["size"], torch.Tensor):
        return float(target["size"][0]), float(target["size"][1])
    return float(fallback), float(fallback)


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

    for (cx, cy, _), poly, txt in zip(pred_centers, pred_polygons, pred_texts):
        pts_flat = [coord for pt in poly for coord in pt]
        draw.polygon(pts_flat, outline=(255, 60, 60))

        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=(255, 0, 0), outline=(255, 0, 0))

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
        f"Prédictions SPTSv2-Hybrid — {image_name}\n"
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

                img_h_real, img_w_real = get_eval_image_size(target, fallback=img_size)

                out_i = out_tensor[i]

                gt_polys, gt_texts = extract_all_gt(
                    target, chars, img_w_real, img_h_real,
                    pad_rec_index=pad_rec_index,
                    no_known_char=no_known_char,
                )

                pred_centers, pred_polys, pred_texts, pred_scores = \
                    extract_all_pred(
                        out_i, chars, img_w_real, img_h_real,
                        bins=bins,
                        padding_bins=padding_bins,
                        category_start_index=category_start_index,
                        text_length=text_length,
                        gt_polys=gt_polys,
                    )

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

                ap = compute_map_center(pred_centers, gt_polys)
                all_map_aps.append(ap)

                pred_text_full = " ".join(pred_texts)
                gt_text_full   = " ".join(gt_texts)

                if gt_text_full:
                    all_cer.append(compute_cer(
                        gt_text_full.lower(), pred_text_full.lower()))
                    all_wer.append(compute_wer(
                        gt_text_full.lower(), pred_text_full.lower()))

                all_predictions.append(pred_text_full)
                all_gts.append(gt_text_full)

                if vis_count < max_vis:
                    try:
                        mask_i = samples.mask[i] if samples.mask is not None else None
                        img_np = tensor_to_image(samples.tensors[i], mask_i)
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
    print("RÉSULTATS DE L'ÉVALUATION — SPTSv2-Hybrid")
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
    bars = plt.bar(metrics, values_norm, color="darkorange")
    plt.title("Métriques d'évaluation — SPTSv2-Hybrid")
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
# Restore checkpoint args
# ─────────────────────────────────────────────

_TRAIN_ARG_KEYS = (
    'pad_rec', 'padding_bins', 'bins', 'chars', 'max_length',
    'pad_rec_index', 'no_known_char', 'pre_norm',
    'enc_layers', 'dec_layers', 'window_size', 'obj_num', 'num_box',
    'dim_feedforward', 'hidden_dim', 'dropout', 'depths', 'nheads',
    'num_queries', 'transformer_type', 'dilation', 'position_embedding',
    'pts_key', 'max_size_test', 'min_size_test',
    # hybrid-specific
    'embed_dim', 'patch_size', 'img_size',
    'freeze_backbone_epochs', 'no_freeze_start',
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
# Build hybrid model skeleton (no backbone ckpt loading)
# ─────────────────────────────────────────────

def build_hybrid_model_for_eval(args):
    """
    Reconstruit l'architecture SPTSv2-Hybrid sans charger les checkpoints
    ResNet/ViT individuels. Les poids seront entièrement fournis par le
    checkpoint hybride chargé ensuite avec load_state_dict.
    """
    from models.vit.hybrid import HybridBackbone
    from models.vit.sptsv2 import SPTSv2
    from models.encoder_decoder import build_transformer

    embed_dim = args.embed_dim

    # ── Créer HybridBackbone sans charger les sous-checkpoints ───────────
    backbone = nn.Module.__new__(HybridBackbone)
    nn.Module.__init__(backbone)
    backbone.embed_dim = embed_dim

    # ResNet18 — architecture seule, les poids viendront du checkpoint hybride
    resnet = tv_models.resnet18(pretrained=False)
    backbone.cnn = nn.Sequential(*list(resnet.children())[:-2])

    # ViT-Small — architecture seule
    vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    backbone.vit = vit
    vit_dim = vit.embed_dim  # 384

    # Couches de fusion
    backbone.proj_cnn = nn.Sequential(
        nn.Conv2d(512, embed_dim, kernel_size=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
    )
    backbone.proj_vit = nn.Linear(vit_dim, embed_dim)
    backbone.pool     = nn.AdaptiveAvgPool2d((14, 14))

    # Attributs lus par SPTSv2
    backbone.num_channels = embed_dim
    backbone.grid_size    = (14, 14)

    # Relier la méthode forward depuis la classe (Python la cherche sur le type)
    backbone.__class__ = HybridBackbone

    # ── Transformer + modèle complet ─────────────────────────────────────
    transformer = build_transformer(args)
    num_classes = args.padding_index + 1
    model       = SPTSv2(backbone, transformer, num_classes)

    return model


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from main_hybrid import get_args_parser

    parser = get_args_parser()
    parser.prog = "SPTSv2-Hybrid evaluation"
    parser.set_defaults(output_dir="./evaluation_hybrid_results")

    # Eval-only flags (not in main_hybrid's parser)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint hybride (.pth)")
    parser.add_argument("--max_vis", type=int, default=20,
                        help="Nombre max de visualisations sauvegardées")
    parser.add_argument("--eval_img_size", type=int, default=None,
                        help="Taille image de repli si target sans orig_size "
                             "(défaut : min_size_test)")

    # resnet_ckpt / vit_ckpt ne sont pas utilisés ici (architecture reconstruite
    # depuis le checkpoint hybride), mais l'argument est requis par get_args_parser
    # → on le rend optionnel avec une valeur vide
    for action in parser._actions:
        if action.dest in ("resnet_ckpt", "vit_ckpt"):
            action.required = False
            action.default  = ""

    args = parser.parse_args()

    if not args.val_dataset or not args.data_root:
        parser.error("--val_dataset et --data_root sont requis pour l'évaluation")

    print(f"Chargement du checkpoint hybride : {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)

    # Restaurer les hyperparamètres d'entraînement depuis le checkpoint
    merge_checkpoint_args(args, checkpoint)

    # Calculer les index de tokens (category_start_index, etc.)
    args = process_args(args)

    if args.eval_img_size is None:
        args.eval_img_size = args.min_size_test

    print(f"Test resize : max_size_test={args.max_size_test}, "
          f"min_size_test={args.min_size_test}")
    print(f"Token indices : category_start={args.category_start_index}, "
          f"end={args.end_index}, start={args.start_index}, "
          f"padding_bins={args.padding_bins}, pad_rec={args.pad_rec}, "
          f"pad_rec_index={args.pad_rec_index}, no_known_char={args.no_known_char}")
    print(f"embed_dim={args.embed_dim}  hidden_dim={args.hidden_dim}")

    # ── Construction du modèle hybride ──────────────────────────────────
    model = build_hybrid_model_for_eval(args)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        missing, unexpected = model.load_state_dict(
            checkpoint["model"], strict=False
        )
        epoch = checkpoint.get("epoch", "?")
        print(f"Checkpoint hybride chargé — epoch {epoch}")
        if missing:
            print(f"  Clés manquantes ({len(missing)}) : {missing[:5]} ...")
        if unexpected:
            print(f"  Clés inattendues ({len(unexpected)}) : {unexpected[:5]} ...")
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print("State dict direct chargé")
        if missing:
            print(f"  Clés manquantes ({len(missing)}) : {missing[:5]} ...")
        if unexpected:
            print(f"  Clés inattendues ({len(unexpected)}) : {unexpected[:5]} ...")

    device = torch.device(args.device)
    model.to(device)
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
        device               = device,
        chars                = args.chars,
        start_index          = args.start_index,
        output_dir           = args.output_dir,
        bins                 = args.bins,
        padding_bins         = args.padding_bins,
        category_start_index = args.category_start_index,
        text_length          = args.max_length,
        img_size             = args.eval_img_size,
        max_vis              = args.max_vis,
        pad_rec_index        = args.pad_rec_index,
        no_known_char        = args.no_known_char,
    )

    save_results(results, args.output_dir)
