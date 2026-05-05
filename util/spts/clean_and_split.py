"""
clean_and_split.py
==================
Clean a COCO-style spts JSON (produced by build_annotations_json_ic19.py or
build_icdar_gt_json.py), then split it into train and val sets.

Cleaning
--------
1. Drop annotations whose ``rec_string`` contains any character outside the
   95-char spts charset (space → ~, same as main_original.py --chars).
   "###" (ICDAR don't-care) is also dropped.
2. Drop images that have no annotations left after step 1.
3. Copy surviving images to the output folder (always on).

Split
-----
Images are shuffled with a fixed seed and split into train / val according to
``--val-ratio`` (default 0.2 → 80 % train / 20 % val).
Two sub-directories are created under ``--out-dir``:
  <out-dir>/train/  → train.json  +  train_images/
  <out-dir>/val/    → val.json    +  val_images/

Usage
-----
  # default: E:/Nassila/ICDAR/ICDAR2019_rrc/train.json, 80/20 split
  python util/dataset/clean_and_split.py

  # custom paths
  python util/dataset/clean_and_split.py \\
      --json   E:/Nassila/ICDAR/ICDAR2019_rrc/train.json \\
      --img-dir E:/Nassila/ICDAR/ICDAR2019_rrc/images \\
      --out-dir E:/Nassila/ICDAR/ICDAR2019_rrc/split \\
      --val-ratio 0.2 \\
      --seed 42
"""
# default paths (E:\Nassila\ICDAR\ICDAR2019_rrc\train.json, 80/20 split)
# python util/spts/clean_and_split.py

# # custom
# python util/spts/clean_and_split.py --json  "E:/Nassila/ICDAR/ICDAR2019_rrc/train.json" --img-dir "E:\Nassila\ICDAR\ICDAR2019_rrc\train_full_images_0" --out-dir "E:/Nassila/ICDAR/ICDAR2019_rrc/split"
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Charset — 95-char printable ASCII, matches main_original.py --chars default
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./'
    '0123456789'
    ':;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz'
    '{|}~'
)
_CHARSET_SET: frozenset[str] = frozenset(CHARS)
_DONT_CARE = "###"

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_JSON    = r"E:\Nassila\ICDAR\ICDAR2019_rrc\train.json"
DEFAULT_OUT_DIR = r"E:\Nassila\ICDAR\ICDAR2019_rrc\split"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _text_ok(text: str) -> bool:
    """Return True only if text is not ### and every char is in the charset."""
    if text == _DONT_CARE:
        return False
    return all(ch in _CHARSET_SET for ch in text)


def clean(
    data: dict,
) -> tuple[dict, dict]:
    """Remove out-of-charset / don't-care annotations and imageless images.

    Returns ``(clean_data, stats)``.
    The returned dict keeps the same top-level structure (licenses, info,
    categories, images, annotations).
    """
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    stats = {
        "images_in":         len(data.get("images", [])),
        "annotations_in":    len(data.get("annotations", [])),
        "ann_dropped_charset": 0,
        "images_dropped_empty": 0,
        "images_out":        0,
        "annotations_out":   0,
    }

    clean_images: list[dict] = []
    clean_anns:   list[dict] = []

    for img in data.get("images", []):
        img_id = img["id"]
        raw_anns = anns_by_image.get(img_id, [])

        valid: list[dict] = []
        for ann in raw_anns:
            text = ann.get("rec_string", "")
            if _text_ok(text):
                valid.append(ann)
            else:
                stats["ann_dropped_charset"] += 1

        if not valid:
            stats["images_dropped_empty"] += 1
            continue

        clean_images.append(img)
        clean_anns.extend(valid)

    stats["images_out"]      = len(clean_images)
    stats["annotations_out"] = len(clean_anns)

    result = {
        "licenses":    data.get("licenses", []),
        "info":        data.get("info", {}),
        "categories":  data.get("categories", []),
        "images":      clean_images,
        "annotations": clean_anns,
    }
    return result, stats


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_coco(
    data: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Randomly split *data* into train and val COCO dicts.

    Images are shuffled then partitioned; annotations follow their image.
    Returns ``(train_data, val_data)``.
    """
    images = list(data["images"])
    random.seed(seed)
    random.shuffle(images)

    n_val   = max(1, round(len(images) * val_ratio))
    n_train = len(images) - n_val

    val_imgs   = images[:n_val]
    train_imgs = images[n_val:]

    val_ids   = {img["id"] for img in val_imgs}
    train_ids = {img["id"] for img in train_imgs}

    train_anns: list[dict] = []
    val_anns:   list[dict] = []
    for ann in data["annotations"]:
        if ann["image_id"] in train_ids:
            train_anns.append(ann)
        elif ann["image_id"] in val_ids:
            val_anns.append(ann)

    base = {
        "licenses":   data.get("licenses", []),
        "info":       data.get("info", {}),
        "categories": data.get("categories", []),
    }

    train_data = {**base, "images": train_imgs, "annotations": train_anns}
    val_data   = {**base, "images": val_imgs,   "annotations": val_anns}
    return train_data, val_data


# ---------------------------------------------------------------------------
# Image copying
# ---------------------------------------------------------------------------

def copy_images(
    split_data: dict,
    img_src_dir: Path,
    img_dst_dir: Path,
) -> tuple[int, int]:
    """Copy images referenced in *split_data* from src to dst.

    Returns ``(copied, missing)``.
    """
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    copied = missing = 0
    for img in split_data["images"]:
        src = img_src_dir / img["file_name"]
        dst = img_dst_dir / img["file_name"]
        if src.is_file():
            if not dst.exists():
                shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  [WARN] source image not found: {src}", file=sys.stderr)
            missing += 1
    return copied, missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean and split an spts COCO JSON into train/val sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--json", "-j", default=DEFAULT_JSON, metavar="PATH",
        help=f"Input COCO JSON (default: {DEFAULT_JSON}).",
    )
    p.add_argument(
        "--img-dir", metavar="PATH", default=None,
        help=(
            "Folder containing source images. "
            "Defaults to an 'images' sub-folder next to --json."
        ),
    )
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR, metavar="PATH",
        help=f"Root output directory (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--val-ratio", type=float, default=0.2, metavar="FLOAT",
        help="Fraction of images to use for validation (default: 0.2).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    json_path = Path(args.json).resolve()
    out_dir   = Path(args.out_dir).resolve()

    if args.img_dir:
        img_src_dir = Path(args.img_dir).resolve()
    else:
        img_src_dir = json_path.parent / "images"

    if not json_path.is_file():
        print(f"Error: JSON not found: {json_path}", file=sys.stderr)
        return 2
    if not img_src_dir.is_dir():
        print(f"Error: image directory not found: {img_src_dir}", file=sys.stderr)
        return 2

    print(f"Input JSON : {json_path}")
    print(f"Image dir  : {img_src_dir}")
    print(f"Output dir : {out_dir}")
    print(f"Val ratio  : {args.val_ratio:.0%}  (seed={args.seed})")
    print(f"Charset    : {len(CHARS)} chars (space .. ~), PAD=96")
    print()

    # ── Load ────────────────────────────────────────────────────────────────
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded  : {len(data.get('images',[]))} images, "
          f"{len(data.get('annotations',[]))} annotations")

    # ── Clean ───────────────────────────────────────────────────────────────
    clean_data, cstats = clean(data)
    print()
    print("=== Cleaning ===")
    print(f"  Images in          : {cstats['images_in']}")
    print(f"  Annotations in     : {cstats['annotations_in']}")
    print(f"  Ann dropped (charset / ###) : {cstats['ann_dropped_charset']}")
    print(f"  Images dropped (empty)      : {cstats['images_dropped_empty']}")
    print(f"  Images remaining   : {cstats['images_out']}")
    print(f"  Annotations remain : {cstats['annotations_out']}")

    if cstats["images_out"] == 0:
        print("\n[ERROR] No images survived cleaning — aborting.", file=sys.stderr)
        return 1

    # ── Split ───────────────────────────────────────────────────────────────
    train_data, val_data = split_coco(clean_data, args.val_ratio, args.seed)
    print()
    print("=== Split ===")
    print(f"  Train : {len(train_data['images']):>6} images, "
          f"{len(train_data['annotations']):>7} annotations")
    print(f"  Val   : {len(val_data['images']):>6} images, "
          f"{len(val_data['annotations']):>7} annotations")

    # ── Write JSON ──────────────────────────────────────────────────────────
    train_dir = out_dir / "train"
    val_dir   = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_json = train_dir / "train.json"
    val_json   = val_dir   / "val.json"

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, separators=(",", ":"))
    with val_json.open("w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, separators=(",", ":"))

    # ── Copy images ─────────────────────────────────────────────────────────
    print()
    print("=== Copying images ===")
    train_img_dir = train_dir / "train_images"
    val_img_dir   = val_dir   / "val_images"

    tr_copied, tr_miss = copy_images(train_data, img_src_dir, train_img_dir)
    print(f"  Train images copied : {tr_copied}"
          + (f"  ({tr_miss} missing)" if tr_miss else ""))

    vl_copied, vl_miss = copy_images(val_data, img_src_dir, val_img_dir)
    print(f"  Val   images copied : {vl_copied}"
          + (f"  ({vl_miss} missing)" if vl_miss else ""))

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("=== Output ===")
    print(f"  {train_json}")
    print(f"  {train_img_dir}/")
    print(f"  {val_json}")
    print(f"  {val_img_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
