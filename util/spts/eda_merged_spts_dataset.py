"""
EDA for merged SPTS COCO JSONs (output of merge_spts_datasets.py).

Reports image counts, dimensions, aspect ratios, optional on-disk checks,
per-source breakdown (from file_name prefix), annotation counts, bbox/bezier
stats, text/character analysis (Latin vs digits vs punctuation, etc.),
plus enhanced analyses: spatial distribution, text-bbox correlation,
n-gram analysis, image quality metrics, outlier detection, dataset balance,
and text complexity metrics.

Examples
--------
    python util/spts/eda_merged_spts_dataset.py "D:\\original_icdar2017\\ICDAR2017_spts\\test.json"

    python util/spts/eda_merged_spts_dataset.py D:/merged_train_final/merged_train.json \\
        --images_dir D:/merged_train_final/train_images --plots_dir D:/merged_train_final/eda_plots \\
        --report_txt D:/merged_train_final/eda_report.txt



"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MPL = False

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class _TeeTextStream:
    """Duplicate ``print`` / ``sys.stdout.write`` to several text streams."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

    def isatty(self) -> bool:
        return False


# Same 95-char printable ASCII as merge_spts_datasets.py (for decoding rec if needed)
_DEFAULT_CHARS = (
    ' !"#$%&\'()*+,-./0123456789:;<=>?@'
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    '[\\]^_`'
    "abcdefghijklmnopqrstuvwxyz"
    '{|}~'
)
_PAD_IDX = 96


def _decode_rec(rec: list[Any], chars: str = _DEFAULT_CHARS, pad_idx: int = _PAD_IDX) -> str:
    out: list[str] = []
    for v in rec:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i >= pad_idx:
            break
        if 0 <= i < len(chars):
            out.append(chars[i])
    return "".join(out)


def _text_from_ann(ann: dict, chars: str, pad_idx: int) -> str:
    rs = ann.get("rec_string")
    if rs is not None and str(rs).strip():
        return str(rs)
    return _decode_rec(ann.get("rec", []) or [], chars=chars, pad_idx=pad_idx)


def _dataset_prefix(file_name: str) -> str:
    """Merged images are ``{ds}_{original_stem}.ext`` — take prefix before first '_'."""
    stem = Path(file_name).name
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "unknown"


def _classify_string(s: str) -> str:
    """Coarse label for whole annotation text (ASCII-oriented, matches spts charset)."""
    if not s:
        return "empty"
    stripped = s.strip()
    if not stripped:
        return "whitespace_only"

    has_digit = any(c.isdigit() for c in stripped)
    has_latin = any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in stripped)
    has_other_print = any(
        not c.isdigit() and not (("A" <= c <= "Z") or ("a" <= c <= "z")) and not c.isspace()
        for c in stripped
    )

    if has_digit and not has_latin and not has_other_print:
        return "digits_only"
    if has_latin and not has_digit and not has_other_print:
        return "latin_letters_only"
    if has_latin and has_digit and not has_other_print:
        return "alnum_no_punct"
    if has_latin and has_other_print:
        return "latin_with_punct_or_symbol"
    if has_digit and has_other_print and not has_latin:
        return "digits_with_punct_or_symbol"
    if has_digit and has_latin and has_other_print:
        return "mixed_alnum_punct"
    return "other"


def _char_bucket(c: str) -> str:
    if c.isspace():
        return "whitespace"
    if c.isdigit():
        return "digit"
    if "A" <= c <= "Z":
        return "latin_upper"
    if "a" <= c <= "z":
        return "latin_lower"
    # printable ASCII punctuation / symbols for spts range
    if 32 <= ord(c) <= 126:
        return "ascii_punct_or_symbol"
    return "non_ascii_or_control"


def _percentiles(arr: np.ndarray, ps: tuple[float, ...]) -> dict[float, float]:
    if arr.size == 0:
        return {p: float("nan") for p in ps}
    out: dict[float, float] = {}
    for p in ps:
        out[p] = float(np.percentile(arr, p))
    return out


def _summarize_numeric(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"  {name}: (empty)")
        return
    pct = _percentiles(arr, (0, 5, 25, 50, 75, 95, 100))
    print(
        f"  {name}: n={arr.size}  "
        f"mean={arr.mean():.4g}  std={arr.std():.4g}  "
        f"min={pct[0]:.4g}  p50={pct[50]:.4g}  max={pct[100]:.4g}"
    )
    print(
        f"         p5={pct[5]:.4g}  p25={pct[25]:.4g}  "
        f"p75={pct[75]:.4g}  p95={pct[95]:.4g}"
    )


def _maybe_plot_hist(
    values: np.ndarray,
    title: str,
    xlabel: str,
    out_path: Path | None,
    *,
    bins: int = 60,
    log_y: bool = False,
) -> None:
    if out_path is None or not _HAS_MPL or values.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="#2c5282", edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def add_spatial_distribution(anns: list[dict], images: list[dict], plots_dir: Path | None) -> None:
    """Analyze where annotations appear in images (heatmap of centers)"""
    if plots_dir is None or not _HAS_MPL:
        return
    
    centers_x = []
    centers_y = []
    
    # Create image ID to dimensions mapping
    img_dimensions = {im["id"]: (im.get("width", 0), im.get("height", 0)) for im in images}
    
    for a in anns:
        bb = a.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            x, y, w, h = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            img_id = int(a.get("image_id", -1))
            if img_id in img_dimensions:
                width, height = img_dimensions[img_id]
                if width > 0 and height > 0:
                    centers_x.append((x + w/2) / width)
                    centers_y.append((y + h/2) / height)
    
    if centers_x:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 2D histogram/heatmap
        h2d, xedges, yedges = np.histogram2d(centers_x, centers_y, bins=20)
        im = ax1.imshow(h2d.T, origin='lower', extent=[0,1,0,1], cmap='hot', aspect='auto')
        ax1.set_title("Annotation Center Heatmap")
        ax1.set_xlabel("Normalized X")
        ax1.set_ylabel("Normalized Y")
        plt.colorbar(im, ax=ax1)
        
        # Marginal distributions
        ax2.hist(centers_x, bins=30, alpha=0.5, label="X centers", density=True)
        ax2.hist(centers_y, bins=30, alpha=0.5, label="Y centers", density=True)
        ax2.set_title("Marginal Distributions")
        ax2.set_xlabel("Normalized position")
        ax2.legend()
        
        fig.tight_layout()
        fig.savefig(plots_dir / "spatial_distribution.png", dpi=150)
        plt.close(fig)
        
        print("  Spatial distribution plot saved: spatial_distribution.png")


def add_length_vs_size_correlation(anns: list[dict], chars: str, pad_idx: int, plots_dir: Path | None) -> None:
    """Correlation between text length and bbox dimensions"""
    if plots_dir is None or not _HAS_MPL:
        return
    
    lengths = []
    bbox_areas = []
    bbox_perimeters = []
    
    for a in anns:
        t = _text_from_ann(a, chars=chars, pad_idx=pad_idx)
        bb = a.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4 and t:
            w, h = float(bb[2]), float(bb[3])
            lengths.append(len(t))
            bbox_areas.append(w * h)
            bbox_perimeters.append(2 * (w + h))
    
    if lengths and len(lengths) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        corr_area = np.corrcoef(lengths, bbox_areas)[0,1] if len(lengths) > 1 else 0
        axes[0].scatter(lengths, bbox_areas, alpha=0.3, s=1)
        axes[0].set_xlabel("Text length (chars)")
        axes[0].set_ylabel("BBox area (px²)")
        axes[0].set_title(f"Area vs Length (r={corr_area:.3f})")
        
        corr_perim = np.corrcoef(lengths, bbox_perimeters)[0,1] if len(lengths) > 1 else 0
        axes[1].scatter(lengths, bbox_perimeters, alpha=0.3, s=1)
        axes[1].set_xlabel("Text length (chars)")
        axes[1].set_ylabel("BBox perimeter (px)")
        axes[1].set_title(f"Perimeter vs Length (r={corr_perim:.3f})")
        
        fig.tight_layout()
        fig.savefig(plots_dir / "length_vs_bbox_size.png", dpi=150)
        plt.close(fig)
        
        print(f"  Text-bbox correlation: area r={corr_area:.3f}, perimeter r={corr_perim:.3f}")


def add_ngram_analysis(texts: list[str], plots_dir: Path | None, n: int = 3) -> None:
    """Most common character n-grams (for language patterns)"""
    if plots_dir is None or not _HAS_MPL or not texts:
        return
    
    from collections import defaultdict
    
    ngrams = defaultdict(int)
    for text in texts:
        if len(text) >= n:
            for i in range(len(text) - n + 1):
                ngrams[text[i:i+n]] += 1
    
    if ngrams:
        top_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)[:20]
        labels, counts = zip(*top_ngrams)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.title(f"Top 20 {n}-grams")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plots_dir / f"top_{n}grams.png", dpi=150)
        plt.close()
        
        print(f"\n  Top {n}-grams:")
        for gram, count in top_ngrams[:10]:
            print(f"    '{gram}': {count}")


def add_image_quality_metrics(images_dir: Path | None, images: list[dict], plots_dir: Path | None) -> None:
    """Analyze image sharpness, contrast, compression artifacts"""
    if not _HAS_CV2 or not images_dir or not images_dir.is_dir() or plots_dir is None:
        return
    
    sharpness = []
    contrast = []
    noise_levels = []
    
    # Sample first 100 images for performance
    sample_size = min(100, len(images))
    print(f"  Analyzing image quality on {sample_size} sample images...")
    
    for im in images[:sample_size]:
        fn = str(im.get("file_name", ""))
        p = images_dir / fn
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Laplacian variance for sharpness
                sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                # Contrast (RMS)
                contrast.append(gray.std())
                
                # Simple noise estimate (difference between image and blurred version)
                blurred = cv2.GaussianBlur(gray, (5,5), 0)
                noise = np.abs(gray.astype(float) - blurred.astype(float)).mean()
                noise_levels.append(noise)
    
    if sharpness and _HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(sharpness, bins=30, color='green', alpha=0.7)
        axes[0].set_title(f"Sharpness (Laplacian var)\nmean={np.mean(sharpness):.1f}")
        axes[0].set_xlabel("Variance")
        
        axes[1].hist(contrast, bins=30, color='blue', alpha=0.7)
        axes[1].set_title(f"Contrast (std dev)\nmean={np.mean(contrast):.1f}")
        axes[1].set_xlabel("Pixel value std")
        
        axes[2].hist(noise_levels, bins=30, color='red', alpha=0.7)
        axes[2].set_title(f"Noise estimate\nmean={np.mean(noise_levels):.2f}")
        axes[2].set_xlabel("Mean absolute difference")
        
        fig.tight_layout()
        fig.savefig(plots_dir / "image_quality.png", dpi=150)
        plt.close(fig)
        
        print(f"  Image quality metrics: sharpness={np.mean(sharpness):.1f}, contrast={np.mean(contrast):.1f}")


def add_outlier_report(widths: np.ndarray, heights: np.ndarray, 
                       lengths: np.ndarray, areas: list[float], 
                       plots_dir: Path | None) -> None:
    """Flag potential anomalies in the dataset"""
    outliers = {}
    
    # Use IQR method for outlier detection
    for name, arr in [("width", widths), ("height", heights), 
                      ("text_len", lengths)]:
        if len(arr) > 0:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers[name] = {
                "count": np.sum((arr < lower) | (arr > upper)),
                "percentage": 100 * np.sum((arr < lower) | (arr > upper)) / len(arr),
                "lower_bound": lower,
                "upper_bound": upper
            }
    
    if areas:
        areas_arr = np.array(areas)
        q1, q3 = np.percentile(areas_arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers["bbox_area"] = {
            "count": np.sum((areas_arr < lower) | (areas_arr > upper)),
            "percentage": 100 * np.sum((areas_arr < lower) | (areas_arr > upper)) / len(areas_arr),
            "lower_bound": lower,
            "upper_bound": upper
        }
    
    print("\n## Outlier Detection (IQR method, 1.5x IQR):")
    for name, stats in outliers.items():
        print(f"  {name}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        print(f"    normal range: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    
    if plots_dir and _HAS_MPL:
        # Boxplot of key metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = []
        labels = []
        for name, arr in [("Width", widths), ("Height", heights), 
                         ("Text Len", lengths)]:
            if len(arr) > 0:
                data_to_plot.append(arr)
                labels.append(name)
        
        if areas:
            data_to_plot.append(np.array(areas))
            labels.append("Area")
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        ax.set_yscale('log')
        ax.set_title("Outlier Visualization (log scale)")
        ax.set_ylabel("Value (log scale)")
        fig.tight_layout()
        fig.savefig(plots_dir / "outliers_boxplot.png", dpi=150)
        plt.close(fig)


def add_balance_report(by_ds_img: Counter, ann_by_prefix: Counter, 
                       cat_c: Counter, str_labels: Counter, 
                       plots_dir: Path | None) -> None:
    """Check if dataset is balanced across sources, categories, text types"""
    print("\n## Dataset Balance Analysis:")
    
    # Source balance (images)
    total_imgs = sum(by_ds_img.values())
    print("  Image source distribution:")
    for src, count in by_ds_img.most_common():
        print(f"    {src}: {count} ({100*count/total_imgs:.1f}%)")
    
    # Annotation source balance
    total_anns = sum(ann_by_prefix.values())
    print("\n  Annotation source distribution:")
    for src, count in ann_by_prefix.most_common():
        print(f"    {src}: {count} ({100*count/total_anns:.1f}%)")
    
    # Category balance
    total_cat = sum(cat_c.values())
    print("\n  Category distribution:")
    for cat, count in cat_c.most_common():
        print(f"    {cat}: {count} ({100*count/total_cat:.1f}%)")
    
    # Text type diversity (Shannon entropy)
    if str_labels:
        probs = np.array(list(str_labels.values())) / sum(str_labels.values())
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(str_labels))
        print(f"\n  Text type diversity (Shannon entropy): {entropy:.3f} / {max_entropy:.3f}")
        print(f"  Normalized entropy: {entropy/max_entropy:.3f}")
    
    if plots_dir and _HAS_MPL:
        # Pie chart for source distribution
        if len(by_ds_img) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.pie(by_ds_img.values(), labels=by_ds_img.keys(), autopct='%1.1f%%')
            ax1.set_title("Image Source Distribution")
            
            if len(ann_by_prefix) > 0:
                ax2.pie(ann_by_prefix.values(), labels=ann_by_prefix.keys(), autopct='%1.1f%%')
                ax2.set_title("Annotation Source Distribution")
            
            fig.tight_layout()
            fig.savefig(plots_dir / "source_balance.png", dpi=150)
            plt.close(fig)


def add_text_complexity(texts: list[str], plots_dir: Path | None) -> None:
    """Measure text complexity: unique characters, repetitiveness, etc."""
    if not texts:
        return
    
    unique_char_ratio = []
    repetitiveness = []  # 1 - (unique chars / total chars)
    
    for t in texts:
        if t:
            unique = len(set(t))
            total = len(t)
            unique_char_ratio.append(unique / total)
            repetitiveness.append(1 - (unique / total))
    
    print("\n## Text Complexity Metrics:")
    if unique_char_ratio:
        print(f"  Avg unique char ratio: {np.mean(unique_char_ratio):.3f}")
        print(f"  Std unique char ratio: {np.std(unique_char_ratio):.3f}")
        print(f"  Avg repetitiveness: {np.mean(repetitiveness):.3f}")
        print(f"  (1 = all same char, 0 = all unique)")
    
    if plots_dir and _HAS_MPL and unique_char_ratio:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(unique_char_ratio, bins=50, alpha=0.7, color='purple')
        ax.set_title("Character Diversity Distribution")
        ax.set_xlabel("Unique characters / total characters")
        ax.set_ylabel("Frequency")
        ax.axvline(np.mean(unique_char_ratio), color='red', linestyle='--', 
                  label=f"Mean: {np.mean(unique_char_ratio):.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "text_complexity.png", dpi=150)
        plt.close(fig)


def run_eda(
    json_path: Path,
    images_dir: Path | None,
    chars: str,
    pad_idx: int,
    plots_dir: Path | None,
) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images: list[dict] = data.get("images") or []
    anns: list[dict] = data.get("annotations") or []

    print("=" * 72)
    print(f"EDA: {json_path.resolve()}")
    print("=" * 72)

    # --- Images ---
    n_img = len(images)
    widths = np.array([int(im.get("width", 0) or 0) for im in images], dtype=np.float64)
    heights = np.array([int(im.get("height", 0) or 0) for im in images], dtype=np.float64)
    valid_wh = (widths > 0) & (heights > 0)
    megapixels = (widths[valid_wh] * heights[valid_wh]) / 1e6
    aspect = np.where(heights[valid_wh] > 0, widths[valid_wh] / heights[valid_wh], np.nan)

    print("\n## Images")
    print(f"  Total images (JSON): {n_img}")
    print(f"  Images with width>0 and height>0: {int(valid_wh.sum())}")
    _summarize_numeric("Width (px, from JSON)", widths[valid_wh])
    _summarize_numeric("Height (px, from JSON)", heights[valid_wh])
    _summarize_numeric("Aspect ratio (W/H)", aspect[~np.isnan(aspect)])
    _summarize_numeric("Megapixels (W*H/1e6)", megapixels)

    by_ds_img: Counter[str] = Counter()
    for im in images:
        by_ds_img[_dataset_prefix(str(im.get("file_name", "")))] += 1
    print("\n  Images by merged prefix (first token before '_'):")
    for k, v in by_ds_img.most_common():
        print(f"    {k}: {v}")

    missing_on_disk = 0
    wh_mismatch = 0
    disk_w: list[float] = []
    disk_h: list[float] = []

    if images_dir and images_dir.is_dir():
        print(f"\n  On-disk check (--images_dir): {images_dir.resolve()}")
        if not _HAS_CV2:
            print("    (opencv not importable; skipping read dimensions)")
        for im in images:
            fn = str(im.get("file_name", ""))
            p = images_dir / fn
            if not p.is_file():
                missing_on_disk += 1
                continue
            if _HAS_CV2:
                arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    missing_on_disk += 1
                    continue
                h, w = arr.shape[:2]
                disk_w.append(float(w))
                disk_h.append(float(h))
                jw, jh = int(im.get("width", 0) or 0), int(im.get("height", 0) or 0)
                if jw and jh and (jw != w or jh != h):
                    wh_mismatch += 1
        print(f"    Files missing or unreadable: {missing_on_disk}")
        print(f"    JSON vs disk dimension mismatches (when both set): {wh_mismatch}")
        if disk_w:
            dw = np.array(disk_w)
            dh = np.array(disk_h)
            _summarize_numeric("Width (px, from disk)", dw)
            _summarize_numeric("Height (px, from disk)", dh)

    # --- Annotations linkage ---
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in anns:
        anns_by_image[int(a.get("image_id", -1))].append(a)

    img_ids = {int(im["id"]) for im in images}
    orphan_anns = sum(1 for a in anns if int(a.get("image_id", -1)) not in img_ids)
    imgs_with_zero_ann = sum(1 for im in images if len(anns_by_image[int(im["id"])]) == 0)

    counts_per_image = np.array(
        [len(anns_by_image[int(im["id"])]) for im in images], dtype=np.float64
    )

    print("\n## Annotations (COCO linkage)")
    print(f"  Total annotations: {len(anns)}")
    print(f"  Annotations whose image_id is not in images[]: {orphan_anns}")
    print(f"  Images with zero annotations: {imgs_with_zero_ann}")
    _summarize_numeric("Annotations per image", counts_per_image)

    _maybe_plot_hist(
        counts_per_image,
        "Annotations per image",
        "count",
        plots_dir / "anns_per_image.png" if plots_dir else None,
        bins=min(80, max(20, int(counts_per_image.max()) + 1) if counts_per_image.size else 20),
        log_y=True,
    )
    _maybe_plot_hist(
        np.log10(widths[valid_wh] + 1),
        "log10(width+1) from JSON",
        "log10(px)",
        plots_dir / "log_width.png" if plots_dir else None,
    )
    _maybe_plot_hist(
        np.log10(heights[valid_wh] + 1),
        "log10(height+1) from JSON",
        "log10(px)",
        plots_dir / "log_height.png" if plots_dir else None,
    )

    # --- Bboxes ---
    bbox_w: list[float] = []
    bbox_h: list[float] = []
    areas: list[float] = []
    for a in anns:
        bb = a.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            _, _, bw, bh = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            bbox_w.append(bw)
            bbox_h.append(bh)
        ar = a.get("area")
        if ar is not None:
            try:
                areas.append(float(ar))
            except (TypeError, ValueError):
                pass

    print("\n## Bounding boxes (annotation bbox [x,y,w,h])")
    if bbox_w:
        bw = np.array(bbox_w)
        bh = np.array(bbox_h)
        _summarize_numeric("BBox width", bw)
        _summarize_numeric("BBox height", bh)
        asp_bb = np.where(bh > 0, bw / bh, np.nan)
        _summarize_numeric("BBox aspect ratio (w/h)", asp_bb[~np.isnan(asp_bb)])
    if areas:
        _summarize_numeric("Annotation area field", np.array(areas))

    iscrowd_c = Counter(int(a.get("iscrowd", 0) or 0) for a in anns)
    cat_c = Counter(int(a.get("category_id", -1)) for a in anns)
    print("\n## Annotation flags / categories")
    print("  iscrowd (value -> count):", dict(iscrowd_c.most_common()))
    print("  category_id (value -> count):", dict(cat_c.most_common()))

    rec_lens: list[int] = []
    for a in anns:
        r = a.get("rec")
        if isinstance(r, list):
            # count indices before pad (same rule as decode)
            n = 0
            for v in r:
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv >= pad_idx:
                    break
                n += 1
            rec_lens.append(n)
    if rec_lens:
        _summarize_numeric("rec list length (indices before pad_idx)", np.array(rec_lens, dtype=np.float64))

    # --- Bezier ---
    n_bezier = 0
    bezier_lens: Counter[int] = Counter()
    for a in anns:
        bp = a.get("bezier_pts")
        if isinstance(bp, list):
            n_bezier += 1
            bezier_lens[len(bp)] += 1
    print("\n## Bezier control points")
    print(f"  Annotations with list bezier_pts: {n_bezier}")
    if bezier_lens:
        print("  Length of bezier_pts list (value -> count):")
        for L, c in sorted(bezier_lens.items()):
            print(f"    {L}: {c}")

    # --- Text / characters ---
    texts: list[str] = []
    lengths: list[int] = []
    str_labels: Counter[str] = Counter()
    char_bucket_total: Counter[str] = Counter()
    char_freq: Counter[str] = Counter()
    word_counts: list[int] = []

    for a in anns:
        t = _text_from_ann(a, chars=chars, pad_idx=pad_idx)
        texts.append(t)
        lengths.append(len(t))
        str_labels[_classify_string(t)] += 1
        for c in t:
            char_bucket_total[_char_bucket(c)] += 1
            char_freq[c] += 1
        # simple "word" = split on whitespace, non-empty tokens
        words = [w for w in re.split(r"\s+", t.strip()) if w] if t.strip() else []
        word_counts.append(len(words))

    lens_arr = np.array(lengths, dtype=np.float64)
    wc_arr = np.array(word_counts, dtype=np.float64)

    print("\n## Text (rec_string or decoded rec)")
    print(f"  Non-empty strings: {sum(1 for t in texts if t.strip())}")
    print(f"  Empty / missing: {sum(1 for t in texts if not t)}")
    _summarize_numeric("Characters per annotation", lens_arr)
    _summarize_numeric("Whitespace-separated tokens per annotation", wc_arr)

    print("\n  Whole-string coarse labels (ASCII-oriented):")
    total_s = sum(str_labels.values()) or 1
    for lab, c in str_labels.most_common():
        print(f"    {lab}: {c}  ({100.0 * c / total_s:.2f}%)")

    print("\n  Character-level buckets (over all characters in all strings):")
    total_ch = sum(char_bucket_total.values()) or 1
    for lab, c in char_bucket_total.most_common():
        print(f"    {lab}: {c}  ({100.0 * c / total_ch:.2f}%)")

    # Latin vs digit emphasis
    latin_chars = char_bucket_total["latin_upper"] + char_bucket_total["latin_lower"]
    digit_chars = char_bucket_total["digit"]
    print("\n  Latin letters (A-Z + a-z) vs digits (character counts):")
    print(f"    latin: {int(latin_chars)}  ({100.0 * latin_chars / total_ch:.2f}%)")
    print(f"    digit: {int(digit_chars)}  ({100.0 * digit_chars / total_ch:.2f}%)")

    print("\n  Top 30 characters (glyph -> count):")
    for ch, c in char_freq.most_common(30):
        disp = repr(ch) if ch in "\n\r\t" or ord(ch) < 32 else ch
        print(f"    {disp}: {c}")

    _maybe_plot_hist(
        lens_arr,
        "Characters per annotation",
        "chars",
        plots_dir / "chars_per_ann.png" if plots_dir else None,
        bins=80,
        log_y=True,
    )

    # --- Per source (prefix) annotations ---
    ann_by_prefix: Counter[str] = Counter()
    id_to_fn = {int(im["id"]): str(im.get("file_name", "")) for im in images}
    for a in anns:
        iid = int(a.get("image_id", -1))
        fn = id_to_fn.get(iid, "")
        ann_by_prefix[_dataset_prefix(fn)] += 1

    print("\n## Annotations by image filename prefix (merged dataset tag)")
    for k, v in ann_by_prefix.most_common():
        print(f"    {k}: {v}")

    # --- Enhanced Analyses ---
    print("\n" + "=" * 72)
    print("ENHANCED ANALYSES")
    print("=" * 72)
    
    # 1. Spatial distribution
    print("\n## Spatial Distribution Analysis")
    add_spatial_distribution(anns, images, plots_dir)
    
    # 2. Text-bbox correlation
    print("\n## Text-BBox Correlation Analysis")
    add_length_vs_size_correlation(anns, chars, pad_idx, plots_dir)
    
    # 3. N-gram analysis
    print("\n## N-gram Analysis")
    add_ngram_analysis(texts, plots_dir, n=3)
    
    # 4. Image quality metrics
    print("\n## Image Quality Analysis")
    add_image_quality_metrics(images_dir, images, plots_dir)
    
    # 5. Outlier report
    print("\n## Outlier Detection")
    add_outlier_report(widths[valid_wh], heights[valid_wh], lens_arr, areas, plots_dir)
    
    # 6. Dataset balance
    print("\n## Dataset Balance")
    add_balance_report(by_ds_img, ann_by_prefix, cat_c, str_labels, plots_dir)
    
    # 7. Text complexity
    print("\n## Text Complexity")
    add_text_complexity(texts, plots_dir)
    
    if plots_dir and _HAS_MPL:
        print(f"\nAll plots written under: {plots_dir.resolve()}")
    elif plots_dir and not _HAS_MPL:
        print("\n(matplotlib not available; skipping plots)")
    print("\n" + "=" * 72)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EDA for merged SPTS COCO JSON (merge_spts_datasets.py output).",
    )
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to merged_train.json or merged_test.json",
    )
    p.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="Folder of merged images (train_images / test_images) for on-disk checks.",
    )
    p.add_argument(
        "--plots_dir",
        type=Path,
        default=None,
        help="If set and matplotlib is installed, save histogram PNGs here.",
    )
    p.add_argument(
        "--chars",
        default=_DEFAULT_CHARS,
        help="Charset for decoding rec when rec_string is empty (default: 95 printable ASCII).",
    )
    p.add_argument(
        "--pad_idx",
        type=int,
        default=_PAD_IDX,
        help="Padding index in rec lists (default: 96).",
    )
    p.add_argument(
        "--report_txt",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the same text report to this UTF-8 .txt file (console output unchanged).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    jp = args.json_path
    if not jp.is_file():
        raise SystemExit(f"JSON not found: {jp}")

    report_path: Path | None = args.report_txt
    report_fh = None
    old_stdout = sys.stdout
    if report_path is not None:
        report_path = report_path.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_fh = open(report_path, "w", encoding="utf-8", newline="\n")
        sys.stdout = _TeeTextStream(old_stdout, report_fh)

    try:
        run_eda(
            jp,
            images_dir=args.images_dir,
            chars=str(args.chars),
            pad_idx=int(args.pad_idx),
            plots_dir=args.plots_dir,
        )
    finally:
        if report_fh is not None:
            sys.stdout = old_stdout
            report_fh.close()
            print(f"Report saved to: {report_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
