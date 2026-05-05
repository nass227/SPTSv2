"""
ICDAR2019 → SPTSv2-compatible dataset pipeline
===============================================
Chains in one command:
  1. EDA on the raw ICDAR2019 GT JSON  (before any changes)
  2. Clean annotations                 (delete mode + remove ###)
  3. Build COCO-style JSON             (bezier_pts + rec, 95-char SPTSv2 charset)
  4. EDA on the final COCO JSON        (after build)
  5. Split into train / test sets

The charset and PAD index match main.py defaults exactly:
  --chars   ' !"#$%&...~'  (95 printable ASCII chars including space)
  PAD_IDX = 96              (--pad_rec_index default)
  OOV_IDX = 95              (--no_known_char default)

Usage:
  python build_icdar2019_pipeline.py \
      --json-input  path/to/train_full_labels.json \
      --img-dir     path/to/images \
      --output-dir  path/to/output \
      [--train-ratio 0.8] \
      [--seed 42] \
      [--copy-images] \
      [--rec-string]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Character set — MUST match main.py --chars default exactly
#
# 95 printable ASCII chars: space (0x20) through tilde (0x7E)
# Index layout:
#   0  : ' '  (space)
#   1  : '!'
#   ...
#   94 : '~'
#   95 : OOV token  (--no_known_char)
#   96 : PAD token  (--pad_rec_index)
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./'           # 0-16  (space + 16 punctuation/digits)
    '0123456789'                   # 17-26
    ':;<=>?@'                      # 27-33
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 34-59
    '[\\]^_`'                      # 60-65
    'abcdefghijklmnopqrstuvwxyz'   # 66-91
    '{|}~'                         # 92-95 → total 95 chars, indices 0-94
)
assert len(CHARS) == 95, f"Charset length must be 95, got {len(CHARS)}"

VALID_CHARS: set[str] = set(CHARS)
OOV_IDX: int = 95   # --no_known_char
PAD_IDX: int = 96   # --pad_rec_index
MAX_LEN: int = 25

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".TIFF"]


# ===========================================================================
# 1. SHARED HELPERS
# ===========================================================================

def _sep(title: str = "", width: int = 64) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'='*2} {title} {'='*pad}")
    else:
        print("=" * width)


def find_image(img_dir: Path, key: str) -> Optional[Path]:
    """Try common prefix/suffix strategies to match a GT key to an image file."""
    def _exact(stem: str) -> Optional[Path]:
        for ext in IMAGE_EXTENSIONS:
            p = img_dir / f"{stem}{ext}"
            if p.is_file():
                return p
        return None

    result = _exact(key)
    if result:
        return result

    if key.startswith("gt_"):
        inner = key[3:]
        for pre in ("", "img_", "image_"):
            result = _exact(pre + inner)
            if result:
                return result
    else:
        result = _exact("gt_" + key)
        if result:
            return result

    for pre in ("", "img_", "image_", "gt_", "gt_img_"):
        for suf in ("", "_img", "_image"):
            result = _exact(pre + key + suf)
            if result:
                return result

    m = re.match(r"^\d+_(.+)$", key)
    if m:
        return find_image(img_dir, m.group(1))

    return None


def get_image_size(img_path: Path) -> tuple[int, int]:
    with PILImage.open(img_path) as im:
        return im.width, im.height


def bezier_from_aabb(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    x_a = x1 + (x2 - x1) / 3
    x_b = x1 + 2 * (x2 - x1) / 3
    return [
        x1, y1,  x_a, y1,  x_b, y1,  x2, y1,
        x2, y2,  x_b, y2,  x_a, y2,  x1, y2,
    ]


def encode_text(text: str, max_len: int = MAX_LEN) -> list[int]:
    """Encode text using the SPTSv2-compatible 95-char charset.

    - Known chars   → their index in CHARS (0-94)
    - Unknown chars → OOV_IDX (95)
    - Padding       → PAD_IDX (96)  to fill up to max_len
    """
    rec: list[int] = []
    for ch in text:
        if len(rec) >= max_len:
            break
        idx = CHARS.find(ch)
        rec.append(idx if idx != -1 else OOV_IDX)
    while len(rec) < max_len:
        rec.append(PAD_IDX)
    return rec


def _text_len(rec: list[int]) -> int:
    """Number of non-PAD tokens in a rec list."""
    return sum(1 for x in rec if x != PAD_IDX)


# ===========================================================================
# 2. EDA — RAW GT JSON  (before cleaning)
# ===========================================================================

def eda_raw(data: dict, title: str = "RAW GT JSON") -> None:
    _sep(f"EDA — {title}")

    all_texts: list[str] = []
    triple_hash = 0
    invalid_char_counter: Counter = Counter()
    text_len_counter: Counter = Counter()
    ann_per_image: list[int] = []

    for key, anns in data.items():
        if not isinstance(anns, list):
            continue
        ann_per_image.append(len(anns))
        for ann in anns:
            t = ann.get("transcription", "")
            if str(t).strip() == "###":
                triple_hash += 1
            all_texts.append(str(t))
            text_len_counter[min(len(str(t)), 30)] += 1
            for ch in str(t):
                if ch not in VALID_CHARS:
                    invalid_char_counter[ch] += 1

    total_images = len(data)
    total_anns   = sum(ann_per_image)
    invalid_anns = sum(
        1 for anns in data.values() if isinstance(anns, list)
        for ann in anns
        if any(ch not in VALID_CHARS for ch in str(ann.get("transcription", "")))
    )

    print(f"  Images (GT keys)          : {total_images:,}")
    print(f"  Total text instances      : {total_anns:,}")
    print(f"  '###' (illegible)         : {triple_hash:,}  ({triple_hash/max(total_anns,1)*100:.1f}%)")
    print(f"  Instances with OOV chars  : {invalid_anns:,}  ({invalid_anns/max(total_anns,1)*100:.1f}%)")
    if ann_per_image:
        print(f"  Annotations per image     : min={min(ann_per_image)} "
              f"max={max(ann_per_image)} "
              f"mean={sum(ann_per_image)/len(ann_per_image):.1f}")

    print("\n  Text-length distribution (truncated at 30):")
    for length in sorted(text_len_counter):
        bar = "#" * min(40, text_len_counter[length] // max(1, total_anns // 400))
        print(f"    len={length:3d} : {text_len_counter[length]:6,}  {bar}")

    if invalid_char_counter:
        print(f"\n  Top-20 OOV characters (not in SPTSv2 95-char set):")
        for ch, cnt in invalid_char_counter.most_common(20):
            print(f"    {repr(ch):10s}  count={cnt:,}")
    else:
        print("\n  No OOV characters found in raw data.")

    # Most frequent valid chars
    char_counter: Counter = Counter()
    for t in all_texts:
        for ch in t:
            if ch in VALID_CHARS:
                char_counter[ch] += 1
    print("\n  Top-20 most frequent valid characters:")
    for ch, cnt in char_counter.most_common(20):
        print(f"    {repr(ch):6s}  {cnt:,}")


# ===========================================================================
# 3. CLEANING  (adapted from clean_icdar2019.py)
# ===========================================================================

def clean_raw_json(
    data: dict,
    remove_triple_hash: bool = True,
) -> tuple[dict, dict]:
    """Delete annotations that contain OOV characters or are '###'.

    Uses the SPTSv2 95-char CHARS set (not the 130-char extended set).
    Returns (cleaned_data, stats).
    """
    cleaned: dict = {}
    stats = {
        "total": 0,
        "valid": 0,
        "deleted_oov": 0,
        "deleted_hash": 0,
        "images_emptied": 0,
    }

    for key, anns in data.items():
        if not isinstance(anns, list):
            cleaned[key] = anns
            continue

        kept = []
        for ann in anns:
            stats["total"] += 1
            t = str(ann.get("transcription", ""))

            if remove_triple_hash and t.strip() == "###":
                stats["deleted_hash"] += 1
                continue

            oov = [ch for ch in t if ch not in VALID_CHARS]
            if oov:
                stats["deleted_oov"] += 1
                continue

            stats["valid"] += 1
            kept.append(ann)

        if kept:
            cleaned[key] = kept
        else:
            stats["images_emptied"] += 1

    return cleaned, stats


def print_clean_stats(stats: dict) -> None:
    _sep("CLEANING STATISTICS")
    print(f"  Total annotations          : {stats['total']:,}")
    print(f"  Kept (valid)               : {stats['valid']:,}")
    print(f"  Deleted (### illegible)    : {stats['deleted_hash']:,}")
    print(f"  Deleted (OOV characters)   : {stats['deleted_oov']:,}")
    print(f"  Images emptied (removed)   : {stats['images_emptied']:,}")


# ===========================================================================
# 4. BUILD COCO JSON  (adapted from build_annotations_json_ic19.py)
# ===========================================================================

def build_coco_from_cleaned(
    cleaned: dict,
    img_dir: Path,
    include_rec_string: bool = False,
) -> tuple[dict, dict]:
    """Build COCO JSON from cleaned ICDAR2019-style dict.

    Uses the 95-char SPTSv2 charset; PAD_IDX=96, OOV_IDX=95.
    """
    images_out: list[dict] = []
    annotations_out: list[dict] = []
    image_id = 1
    ann_id   = 1

    stats = {
        "keys_total": len(cleaned),
        "images_matched": 0,
        "images_missing": 0,
        "annotations": 0,
        "skipped_no_transcription": 0,
        "skipped_bad_points": 0,
        "skipped_zero_area": 0,
        "oov_tokens_used": 0,
    }

    print(f"\n  Processing {stats['keys_total']:,} cleaned GT keys …")

    for key, ann_list in cleaned.items():
        img_path = find_image(img_dir, key)
        if img_path is None:
            stats["images_missing"] += 1
            continue

        try:
            width, height = get_image_size(img_path)
        except Exception as e:
            print(f"  [WARN] cannot open {img_path}: {e}", file=sys.stderr)
            stats["images_missing"] += 1
            continue

        images_out.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        stats["images_matched"] += 1

        if not isinstance(ann_list, list):
            image_id += 1
            continue

        for ann in ann_list:
            if not isinstance(ann, dict):
                continue

            transcription = ann.get("transcription") or ann.get("text") or ann.get("label")
            if not transcription:
                stats["skipped_no_transcription"] += 1
                continue

            points = ann.get("points") or ann.get("bbox_points") or []
            if not points:
                stats["skipped_bad_points"] += 1
                continue

            # Normalise flat [x1,y1,x2,y2,...] → [[x,y],...]
            if points and not isinstance(points[0], (list, tuple)):
                if len(points) >= 4 and len(points) % 2 == 0:
                    points = [[points[i], points[i + 1]] for i in range(0, len(points), 2)]
                else:
                    stats["skipped_bad_points"] += 1
                    continue

            if len(points) < 2:
                stats["skipped_bad_points"] += 1
                continue

            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(float(width), x2);  y2 = min(float(height), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stats["skipped_zero_area"] += 1
                continue

            text = str(transcription)
            rec  = encode_text(text, MAX_LEN)
            stats["oov_tokens_used"] += rec.count(OOV_IDX)

            ann_out: dict = {
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [x1, y1, bw, bh],
                "area":        bw * bh,
                "iscrowd":     0,
                "bezier_pts":  bezier_from_aabb(x1, y1, x2, y2),
                "rec":         rec,
            }
            if include_rec_string:
                ann_out["rec_string"] = text

            annotations_out.append(ann_out)
            ann_id  += 1
            stats["annotations"] += 1

        image_id += 1

    coco = {
        "images":      images_out,
        "annotations": annotations_out,
        "categories":  [{"id": 1, "name": "text"}],
    }
    return coco, stats


def print_build_stats(stats: dict) -> None:
    _sep("BUILD STATISTICS")
    print(f"  GT keys total              : {stats['keys_total']:,}")
    print(f"  Images matched             : {stats['images_matched']:,}")
    print(f"  Images missing (no file)   : {stats['images_missing']:,}")
    print(f"  Annotations written        : {stats['annotations']:,}")
    skipped = (stats["skipped_no_transcription"]
               + stats["skipped_bad_points"]
               + stats["skipped_zero_area"])
    print(f"  Annotations skipped        : {skipped:,}")
    print(f"    └ no transcription       : {stats['skipped_no_transcription']:,}")
    print(f"    └ bad/missing points     : {stats['skipped_bad_points']:,}")
    print(f"    └ zero-area bbox         : {stats['skipped_zero_area']:,}")
    print(f"  OOV tokens in rec          : {stats['oov_tokens_used']:,}  "
          f"(index {OOV_IDX} — char survived cleaning but not in charset)")


# ===========================================================================
# 5. EDA — FINAL COCO JSON  (after build)
# ===========================================================================

def eda_coco(coco: dict, title: str = "FINAL COCO JSON") -> None:
    _sep(f"EDA — {title}")

    images      = coco.get("images", [])
    annotations = coco.get("annotations", [])

    if not annotations:
        print("  No annotations — nothing to analyse.")
        return

    # Annotations per image
    anns_by_img: dict[int, int] = defaultdict(int)
    for ann in annotations:
        anns_by_img[ann["image_id"]] += 1

    counts = list(anns_by_img.values())
    print(f"  Images                     : {len(images):,}")
    print(f"  Annotations                : {len(annotations):,}")
    print(f"  Annotations per image      : min={min(counts)} "
          f"max={max(counts)} "
          f"mean={sum(counts)/len(counts):.1f}")

    # rec token distribution
    rec_len_counter: Counter = Counter()
    idx_counter: Counter     = Counter()
    for ann in annotations:
        rec = ann.get("rec", [])
        non_pad = _text_len(rec)
        rec_len_counter[non_pad] += 1
        for tok in rec:
            idx_counter[tok] += 1

    print(f"\n  rec token counts:")
    print(f"    PAD (index {PAD_IDX}) tokens  : {idx_counter[PAD_IDX]:,}")
    print(f"    OOV (index {OOV_IDX}) tokens  : {idx_counter[OOV_IDX]:,}")
    total_char_tokens = sum(v for k, v in idx_counter.items()
                            if k != PAD_IDX and k != OOV_IDX)
    print(f"    Known char tokens         : {total_char_tokens:,}")

    print("\n  Non-PAD text-length distribution (chars per annotation):")
    for length in sorted(rec_len_counter):
        bar = "#" * min(50, rec_len_counter[length] // max(1, len(annotations) // 500))
        print(f"    len={length:3d} : {rec_len_counter[length]:6,}  {bar}")

    # Image size distribution
    widths  = [img["width"]  for img in images]
    heights = [img["height"] for img in images]
    print(f"\n  Image width  : min={min(widths)}  max={max(widths)}  "
          f"mean={sum(widths)//len(widths)}")
    print(f"  Image height : min={min(heights)}  max={max(heights)}  "
          f"mean={sum(heights)//len(heights)}")

    # Bbox area distribution (buckets)
    areas = [ann["area"] for ann in annotations]
    buckets = [0, 500, 2000, 10000, 50000, float("inf")]
    labels  = ["<500", "500-2k", "2k-10k", "10k-50k", ">50k"]
    area_counts = [0] * len(labels)
    for a in areas:
        for i, (lo, hi) in enumerate(zip(buckets, buckets[1:])):
            if lo <= a < hi:
                area_counts[i] += 1
                break
    print("\n  BBox area distribution:")
    for label, cnt in zip(labels, area_counts):
        bar = "#" * min(40, cnt // max(1, len(annotations) // 400))
        print(f"    {label:12s} : {cnt:6,}  {bar}")

    # Most frequent decoded characters
    char_freq: Counter = Counter()
    for ann in annotations:
        for tok in ann.get("rec", []):
            if 0 <= tok < len(CHARS):
                char_freq[CHARS[tok]] += 1
    print("\n  Top-20 most frequent characters in corpus:")
    for ch, cnt in char_freq.most_common(20):
        print(f"    {repr(ch):6s}  {cnt:,}")


# ===========================================================================
# 6. SPLIT  (adapted from split_dataset.py)
# ===========================================================================

def split_coco(
    coco: dict,
    img_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy_images: bool = False,
) -> dict:
    """Split a COCO dataset into train / test sets."""
    _sep("SPLITTING DATASET")
    random.seed(seed)

    train_img_dir = output_dir / "train_imgs"
    test_img_dir  = output_dir / "test_imgs"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)

    # Build image-id → filename map; only keep images that have annotations
    annotated_ids: set[str] = {str(a["image_id"]) for a in coco.get("annotations", [])}
    id_to_fn: dict[str, str] = {
        str(img["id"]): img["file_name"]
        for img in coco.get("images", [])
        if str(img["id"]) in annotated_ids
    }

    all_ids = list(id_to_fn.keys())
    random.shuffle(all_ids)
    split_at  = int(len(all_ids) * train_ratio)
    train_ids = set(all_ids[:split_at])
    test_ids  = set(all_ids[split_at:])

    print(f"  Total images with annotations : {len(all_ids):,}")
    print(f"  Train split                   : {len(train_ids):,}  ({train_ratio*100:.0f}%)")
    print(f"  Test split                    : {len(test_ids):,}   ({(1-train_ratio)*100:.0f}%)")
    print(f"  Random seed                   : {seed}")
    print(f"  Copy mode                     : {copy_images}")

    def _copy_or_link(src: Path, dst: Path) -> bool:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if copy_images:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.symlink_to(src)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
            return True
        except OSError as exc:
            print(f"  [WARN] {src.name}: {exc}", file=sys.stderr)
            return False

    train_found: set[str] = set()
    test_found:  set[str] = set()
    missing = 0

    for split_name, id_set, found_set, dst_dir in [
        ("train", train_ids, train_found, train_img_dir),
        ("test",  test_ids,  test_found,  test_img_dir),
    ]:
        print(f"\n  Processing {split_name} images …")
        for img_id in id_set:
            fn = id_to_fn[img_id]
            src = img_dir / fn
            if not src.exists():
                # Try extension fallback
                stem = Path(fn).stem
                src = next(
                    (img_dir / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS
                     if (img_dir / f"{stem}{ext}").exists()),
                    None,
                )
            if src is None:
                missing += 1
                continue
            ok = _copy_or_link(src, dst_dir / src.name)
            if ok:
                found_set.add(img_id)
            else:
                missing += 1

    def _filter(keep: set[str]) -> dict:
        imgs = [i for i in coco["images"]      if str(i["id"])       in keep]
        anns = [a for a in coco["annotations"] if str(a["image_id"]) in keep]
        return {
            "images":      imgs,
            "annotations": anns,
            "categories":  coco.get("categories", [{"id": 1, "name": "text"}]),
        }

    train_coco = _filter(train_found)
    test_coco  = _filter(test_found)

    train_json = output_dir / "train.json"
    test_json  = output_dir / "test.json"
    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with test_json.open("w", encoding="utf-8") as f:
        json.dump(test_coco,  f, ensure_ascii=False)

    info = {
        "charset":          CHARS,
        "charset_len":      len(CHARS),
        "pad_idx":          PAD_IDX,
        "oov_idx":          OOV_IDX,
        "max_len":          MAX_LEN,
        "train_ratio":      train_ratio,
        "seed":             seed,
        "train_images":     len(train_found),
        "test_images":      len(test_found),
        "train_annotations": len(train_coco["annotations"]),
        "test_annotations":  len(test_coco["annotations"]),
        "missing_images":    missing,
        "files": {
            "train_json":      str(train_json),
            "test_json":       str(test_json),
            "train_imgs_dir":  str(train_img_dir),
            "test_imgs_dir":   str(test_img_dir),
        },
    }
    info_path = output_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    _sep("SPLIT COMPLETE")
    print(f"  Train images      : {len(train_found):,}  | annotations: {len(train_coco['annotations']):,}")
    print(f"  Test  images      : {len(test_found):,}  | annotations: {len(test_coco['annotations']):,}")
    print(f"  Missing images    : {missing}")
    print(f"  Output dir        : {output_dir}")
    print(f"  Wrote             : train.json  test.json  dataset_info.json")
    print(f"  Charset           : {len(CHARS)} chars, PAD={PAD_IDX}, OOV={OOV_IDX}")

    return info


# ===========================================================================
# 7. MAIN PIPELINE
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICDAR2019 → SPTSv2-compatible COCO dataset pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json-input",  required=True, metavar="PATH",
                   help="Raw ICDAR2019 GT JSON (gt_key: [{points, transcription}])")
    p.add_argument("--img-dir",     required=True, metavar="PATH",
                   help="Folder containing the source images")
    p.add_argument("--output-dir",  required=True, metavar="PATH",
                   help="Output directory for all pipeline artefacts")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Train split ratio (default: 0.8)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--copy-images", action="store_true",
                   help="Copy images instead of symlinks (needed on Windows without admin)")
    p.add_argument("--rec-string",  action="store_true",
                   help="Store the original text as rec_string in each annotation")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args   = parse_args(argv)
    src    = Path(args.json_input).resolve()
    imgdir = Path(args.img_dir).resolve()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.is_file():
        print(f"Error: JSON not found: {src}", file=sys.stderr); return 2
    if not imgdir.is_dir():
        print(f"Error: image dir not found: {imgdir}", file=sys.stderr); return 2

    # -- Configuration summary ------------------------------------------------
    _sep("PIPELINE CONFIGURATION")
    print(f"  Input JSON   : {src}")
    print(f"  Image dir    : {imgdir}")
    print(f"  Output dir   : {outdir}")
    print(f"  Train ratio  : {args.train_ratio}")
    print(f"  Seed         : {args.seed}")
    print(f"  Copy images  : {args.copy_images}")
    print(f"  Charset      : {len(CHARS)} chars (SPTSv2 95-char, matches main.py --chars)")
    print(f"  PAD index    : {PAD_IDX}  (matches --pad_rec_index default)")
    print(f"  OOV index    : {OOV_IDX}  (matches --no_known_char default)")
    print(f"  Max length   : {MAX_LEN}")

    # -------------------------------------------------------------------------
    # Step 1 — Load raw JSON
    # -------------------------------------------------------------------------
    _sep("STEP 1 — Load raw GT JSON")
    with src.open(encoding="utf-8-sig") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, dict):
        print("Error: expected a JSON object at root", file=sys.stderr); return 2
    print(f"  Loaded {len(raw_data):,} keys from {src.name}")

    # -------------------------------------------------------------------------
    # Step 2 — EDA on raw data (before cleaning)
    # -------------------------------------------------------------------------
    eda_raw(raw_data, title="RAW GT  (before cleaning)")

    # -------------------------------------------------------------------------
    # Step 3 — Clean
    # -------------------------------------------------------------------------
    _sep("STEP 3 — Cleaning")
    print("  Mode: DELETE (remove annotations with OOV chars or '###')")
    cleaned, clean_stats = clean_raw_json(raw_data, remove_triple_hash=True)
    print_clean_stats(clean_stats)

    cleaned_path = outdir / "cleaned_labels.json"
    with cleaned_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)
    print(f"\n  Saved cleaned GT → {cleaned_path}")

    # -------------------------------------------------------------------------
    # Step 4 — Build COCO JSON
    # -------------------------------------------------------------------------
    _sep("STEP 4 — Build COCO annotation JSON")
    coco, build_stats = build_coco_from_cleaned(
        cleaned, imgdir, include_rec_string=args.rec_string
    )
    print_build_stats(build_stats)

    full_coco_path = outdir / "full_annotations.json"
    with full_coco_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"\n  Saved full COCO JSON → {full_coco_path}")

    # -------------------------------------------------------------------------
    # Step 5 — EDA on final COCO JSON (after building)
    # -------------------------------------------------------------------------
    eda_coco(coco, title="FINAL COCO  (after build, before split)")

    # -------------------------------------------------------------------------
    # Step 6 — Split into train / test
    # -------------------------------------------------------------------------
    split_coco(
        coco,
        imgdir,
        outdir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
    )

    _sep("PIPELINE DONE")
    print(f"  Output directory: {outdir}")
    print(f"  Files produced:")
    print(f"    cleaned_labels.json   — intermediate cleaned GT")
    print(f"    full_annotations.json — complete COCO JSON")
    print(f"    train.json            — train split (COCO)")
    print(f"    test.json             — test  split (COCO)")
    print(f"    train_imgs/           — train images")
    print(f"    test_imgs/            — test  images")
    print(f"    dataset_info.json     — charset + split metadata")
    print()
    print("  Training command (add to ocr_dataset.py if needed):")
    print("    --dataset_file ocr")
    print("    --train_dataset ICDAR2019_train")
    print("    --val_dataset   ICDAR2019_test")
    print("    --data_root     <output-dir parent>")
    print("  (train_imgs → ICDAR2019/train_imgs, train.json → ICDAR2019/train.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())