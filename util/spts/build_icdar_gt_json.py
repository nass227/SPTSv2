"""

build_icdar_gt_json.py
======================
Convert a folder of ICDAR-style GT .txt files + an image folder into a
COCO-style annotation JSON that matches the ic13_test.json / ic15_test.json
schema used by this project (spts format).

Key behaviours
--------------
* Charset  : 95-char printable ASCII starting from space (same as the spts
             datasets and main_original.py --chars default).
             PAD_IDX = 96.  Space is index 0, '~' is index 94.

* Filtering :
    - Lines whose text is "###" (ICDAR don't-care marker) are DROPPED.
    - Annotations where ANY character is not in the charset are DROPPED
      entirely (the whole annotation is removed, not just the bad chars).
    - Images that end up with zero valid annotations are DROPPED — they are
      not included in the JSON and their images are NOT copied.

* Bezier    : Computed from the actual quad corners returned by the GT,
              NOT from the axis-aligned bounding box.
              Convention (matches ic13/ic15 spts files):
                Top edge  L->R : P0=top-left,   P1=1/3, P2=2/3, P3=top-right
                Bot edge  R->L : P4=bot-right,  P5=1/3, P6=2/3, P7=bot-left

* Images    : Copied to <output_dir>/images/ (or --img-out) with original
              filenames.  A JSON sibling is written at <output_dir>/ann.json
              (or --output).

Supported GT line formats (auto-detected)
-----------------------------------------
  A. x1,y1,x2,y2,x3,y3,x4,y4,SCRIPT,text    ← ICDAR 2017 / 2019
  B. x1,y1,x2,y2,x3,y3,x4,y4,text           ← ICDAR 2013 / 2015 (no script)
  C. x1,y1,x2,y2,text                        ← AABB, 4 coords only

  For format C the four corners are synthesised:
    top-left=(x1,y1), top-right=(x2,y1),
    bot-right=(x2,y2), bot-left=(x1,y2)

Usage
-----
  python util/dataset/build_icdar_gt_json.py \\
      --gt-dir  Data/ICDAR2017/gt_train \\
      --img-dir Data/ICDAR2017/train_images \\
      --out-dir Data/ICDAR2017/spts_train

  # override charset / max text length if needed
  python util/dataset/build_icdar_gt_json.py \\
      --gt-dir  Data/ICDAR2017/gt_train \\
      --img-dir Data/ICDAR2017/train_images \\
      --out-dir Data/ICDAR2017/spts_train \\
      --max-len 25
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Charset — must match main_original.py / main.py --chars default exactly.
# Index layout: 0=' ', 1='!', ..., 94='~'   PAD_IDX=96
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./0123456789:;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz'
    '{|}~'
)   # 95 characters  (index = ord(char) - 32)
PAD_IDX: int = 96
MAX_LEN: int = 25

_CHARSET_SET: frozenset[str] = frozenset(CHARS)

# ---------------------------------------------------------------------------
# Image file extensions to search for
# ---------------------------------------------------------------------------
_IMAGE_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".GIF",
]

# ICDAR don't-care marker
_DONT_CARE = "###"

# Script-label tokens that appear between the 8 coordinates and the text
# in ICDAR 2017 / 2019 style GT files.
_SCRIPT_RE = re.compile(r'^[A-Za-z][A-Za-z\-_]{0,29}$')


# ---------------------------------------------------------------------------
# Charset helpers
# ---------------------------------------------------------------------------

def text_in_charset(text: str, charset: frozenset[str] = _CHARSET_SET) -> bool:
    """Return True only if every character in *text* is in the charset."""
    return all(ch in charset for ch in text)


def encode_text(
    text: str,
    chars: str = CHARS,
    max_len: int = MAX_LEN,
    pad_idx: int = PAD_IDX,
) -> list[int]:
    """Encode text to a fixed-length integer list (charset indices).

    Characters not in *chars* are skipped (should not happen after filtering).
    The list is right-padded with *pad_idx* to exactly *max_len* elements.
    """
    rec: list[int] = []
    for ch in text:
        if len(rec) >= max_len:
            break
        idx = chars.find(ch)
        if idx != -1:
            rec.append(idx)
    while len(rec) < max_len:
        rec.append(pad_idx)
    return rec


# ---------------------------------------------------------------------------
# Bezier control points from quad corners
# ---------------------------------------------------------------------------

def bezier_from_quad(
    x1: float, y1: float,   # top-left
    x2: float, y2: float,   # top-right
    x3: float, y3: float,   # bottom-right
    x4: float, y4: float,   # bottom-left
) -> list[float]:
    """Return 16 floats representing two cubic bezier rails.

    Top edge  (left -> right) : P0=top-left,   P1, P2, P3=top-right
    Bottom edge (right -> left) : P4=bot-right, P5, P6, P7=bot-left

    For a straight segment, the two inner control points divide it into
    thirds: P_inner1 = P_start + 1/3*(P_end - P_start)
            P_inner2 = P_start + 2/3*(P_end - P_start)
    """
    # top edge: top-left -> top-right
    t_cx1 = x1 + (x2 - x1) / 3
    t_cy1 = y1 + (y2 - y1) / 3
    t_cx2 = x1 + 2 * (x2 - x1) / 3
    t_cy2 = y1 + 2 * (y2 - y1) / 3

    # bottom edge: bot-right -> bot-left
    b_cx1 = x3 + (x4 - x3) / 3
    b_cy1 = y3 + (y4 - y3) / 3
    b_cx2 = x3 + 2 * (x4 - x3) / 3
    b_cy2 = y3 + 2 * (y4 - y3) / 3

    return [
        x1, y1,  t_cx1, t_cy1,  t_cx2, t_cy2,  x2, y2,   # top
        x3, y3,  b_cx1, b_cy1,  b_cx2, b_cy2,  x4, y4,   # bottom
    ]


# ---------------------------------------------------------------------------
# GT line parsing
# ---------------------------------------------------------------------------

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _parse_floats(fields: list[str], n: int) -> Optional[list[float]]:
    if len(fields) < n:
        return None
    nums: list[float] = []
    for f in fields[:n]:
        try:
            nums.append(float(f.strip()))
        except ValueError:
            return None
    return nums


def parse_gt_line(
    raw: str,
) -> Optional[tuple[float, float, float, float, float, float, float, float, str]]:
    """Parse one GT line.

    Returns (x1,y1, x2,y2, x3,y3, x4,y4, text) or None.
    The four points are: top-left, top-right, bottom-right, bottom-left.

    For 4-coord AABB lines the corners are synthesised.
    """
    line = raw.strip()
    if not line:
        return None

    # Try CSV splitting first.
    try:
        fields = list(next(csv.reader([line])))
    except Exception:
        fields = [line]

    fields = [_strip_quotes(f.strip()) for f in fields if f.strip()]
    if not fields:
        return None

    # --- 8-coord quad (with optional leading script label) ---
    if len(fields) >= 9:
        nums = _parse_floats(fields, 8)
        if nums is not None:
            tail = fields[8:]
            # Skip a script-label token (e.g. "Latin", "Arabic")
            if (tail
                    and _SCRIPT_RE.match(tail[0])
                    and not tail[0].lstrip('-').replace('.', '', 1).isdigit()):
                tail = tail[1:]
            text = _strip_quotes(",".join(tail))
            if not text:
                return None
            x1, y1, x2, y2, x3, y3, x4, y4 = nums
            return x1, y1, x2, y2, x3, y3, x4, y4, text

    # --- 4-coord AABB ---
    if len(fields) >= 5:
        nums = _parse_floats(fields, 4)
        if nums is not None:
            x1, y1, x2, y2 = nums
            text = _strip_quotes(",".join(fields[4:]))
            if not text:
                return None
            # Synthesise quad corners from AABB
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, text

    return None


# ---------------------------------------------------------------------------
# Image finding
# ---------------------------------------------------------------------------

def _find_exact(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in _IMAGE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def find_image(img_dir: Path, gt_stem: str) -> Optional[Path]:
    """Try common stem transformations to pair a GT stem with its image."""
    result = _find_exact(img_dir, gt_stem)
    if result:
        return result

    # "gt_img_N" -> try "img_N"
    if gt_stem.startswith("gt_"):
        inner = gt_stem[3:]
        for prefix in ("", "img_", "image_"):
            result = _find_exact(img_dir, prefix + inner)
            if result:
                return result

    # plain stem -> try "gt_" prefix
    if not gt_stem.startswith("gt_"):
        result = _find_exact(img_dir, "gt_" + gt_stem)
        if result:
            return result

    # broader prefix search
    for pre in ("img_", "image_", "gt_", "gt_img_"):
        result = _find_exact(img_dir, pre + gt_stem)
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

CATEGORIES = [
    {
        "id": 1,
        "name": "text",
        "supercategory": "beverage",
        "keypoints": ["mean", "xmin", "x2", "x3", "xmax",
                      "ymin", "y2", "y3", "ymax", "cross"],
    }
]


def build_json(
    gt_dir: Path,
    img_dir: Path,
    out_img_dir: Path,
    *,
    chars: str = CHARS,
    charset_set: frozenset[str] = _CHARSET_SET,
    max_len: int = MAX_LEN,
    pad_idx: int = PAD_IDX,
    copy_images: bool = True,
) -> tuple[dict, dict]:
    """Build the COCO-style JSON dict.

    Returns (coco_dict, stats).
    Images with no valid annotations are excluded entirely.
    """
    images_out: list[dict] = []
    annotations_out: list[dict] = []
    image_id = 1
    ann_id = 1

    stats: dict = {
        "gt_files": 0,
        "images_no_match": 0,
        "images_no_ann": 0,
        "images_kept": 0,
        "annotations_kept": 0,
        "lines_dont_care": 0,
        "lines_bad_parse": 0,
        "lines_bad_bbox": 0,
        "lines_charset_fail": 0,
    }

    gt_files = sorted(
        p for p in gt_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )
    stats["gt_files"] = len(gt_files)
    print(f"[INFO] {len(gt_files)} GT files found in {gt_dir}")

    if copy_images:
        out_img_dir.mkdir(parents=True, exist_ok=True)

    for gt_path in gt_files:
        img_path = find_image(img_dir, gt_path.stem)
        if img_path is None:
            stats["images_no_match"] += 1
            continue

        # Read GT
        try:
            raw = gt_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = gt_path.read_text(encoding="latin-1", errors="replace")

        # Open image (needed for width/height and bbox clamping)
        try:
            with PILImage.open(img_path) as im:
                img_w, img_h = im.width, im.height
        except Exception as exc:
            print(f"  [WARN] Cannot open {img_path}: {exc}", file=sys.stderr)
            continue

        valid_anns: list[dict] = []

        for line in raw.splitlines():
            parsed = parse_gt_line(line)
            if parsed is None:
                if line.strip():
                    stats["lines_bad_parse"] += 1
                continue

            x1, y1, x2, y2, x3, y3, x4, y4, text = parsed

            # --- Don't-care filter ---
            if text == _DONT_CARE:
                stats["lines_dont_care"] += 1
                continue

            # --- Charset filter ---
            if not text_in_charset(text, charset_set):
                stats["lines_charset_fail"] += 1
                continue

            # --- Clamp corners to image bounds ---
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))
            x3 = max(0.0, min(x3, img_w))
            y3 = max(0.0, min(y3, img_h))
            x4 = max(0.0, min(x4, img_w))
            y4 = max(0.0, min(y4, img_h))

            # AABB from quad corners
            xs = [x1, x2, x3, x4]
            ys = [y1, y2, y3, y4]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                stats["lines_bad_bbox"] += 1
                continue

            bezier_pts = bezier_from_quad(x1, y1, x2, y2, x3, y3, x4, y4)
            rec = encode_text(text, chars, max_len, pad_idx)

            valid_anns.append({
                "id": ann_id,          # placeholder; reassigned below
                "image_id": image_id,  # placeholder; reassigned below
                "category_id": 1,
                "bbox": [xmin, ymin, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
                "bezier_pts": bezier_pts,
                "rec": rec,
                "rec_string": text,
            })

        # --- Skip image if no valid annotations remain ---
        if not valid_anns:
            stats["images_no_ann"] += 1
            continue

        # Assign real IDs now that we know the image is kept
        for ann in valid_anns:
            ann["id"] = ann_id
            ann["image_id"] = image_id
            ann_id += 1

        images_out.append({
            "coco_url": "",
            "date_captured": "",
            "file_name": img_path.name,
            "flickr_url": "",
            "id": image_id,
            "license": 0,
            "width": img_w,
            "height": img_h,
        })
        annotations_out.extend(valid_anns)
        stats["images_kept"] += 1
        stats["annotations_kept"] += len(valid_anns)
        image_id += 1

        # Copy image
        if copy_images:
            dst = out_img_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)

    coco = {
        "licenses": [],
        "info": {},
        "categories": list(CATEGORIES),
        "images": images_out,
        "annotations": annotations_out,
    }
    return coco, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a spts-format COCO JSON from ICDAR GT txt files.\n"
            "Filters out ### and non-charset annotations.\n"
            "Images with no valid annotations are not copied."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--gt-dir", required=True,
        help="Folder containing GT .txt files (e.g. Data/ICDAR2017/gt_train).",
    )
    p.add_argument(
        "--img-dir", required=True,
        help="Folder containing the source images.",
    )
    p.add_argument(
        "--out-dir", required=True,
        help=(
            "Output root directory.  The JSON is written as <out-dir>/ann.json "
            "and images are copied to <out-dir>/images/."
        ),
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Override the output JSON path (default: <out-dir>/ann.json).",
    )
    p.add_argument(
        "--img-out", default=None,
        help="Override the output image folder (default: <out-dir>/images).",
    )
    p.add_argument(
        "--max-len", type=int, default=MAX_LEN,
        help=f"Max text length (default {MAX_LEN}).",
    )
    p.add_argument(
        "--no-copy", action="store_true",
        help="Do not copy images; only write the JSON.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    gt_dir  = Path(args.gt_dir).resolve()
    img_dir = Path(args.img_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_json = Path(args.output).resolve() if args.output else out_dir / "ann.json"
    out_imgs = Path(args.img_out).resolve() if args.img_out else out_dir / "images"

    if not gt_dir.is_dir():
        print(f"Error: GT directory not found: {gt_dir}", file=sys.stderr)
        return 2
    if not img_dir.is_dir():
        print(f"Error: Image directory not found: {img_dir}", file=sys.stderr)
        return 2

    print(f"GT dir     : {gt_dir}")
    print(f"Image dir  : {img_dir}")
    print(f"Output JSON: {out_json}")
    print(f"Output imgs: {out_imgs}")
    print(f"Charset    : {len(CHARS)} chars, PAD_IDX={PAD_IDX}, max_len={args.max_len}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    coco, stats = build_json(
        gt_dir, img_dir, out_imgs,
        chars=CHARS,
        charset_set=_CHARSET_SET,
        max_len=args.max_len,
        pad_idx=PAD_IDX,
        copy_images=not args.no_copy,
    )

    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False, separators=(",", ":"))

    print()
    print("=" * 55)
    print("Build summary")
    print("=" * 55)
    print(f"  GT files found              : {stats['gt_files']}")
    print(f"  Images with no matching img : {stats['images_no_match']}")
    print(f"  Images kept (have valid ann): {stats['images_kept']}")
    print(f"  Images dropped (no ann left): {stats['images_no_ann']}")
    print(f"  Annotations kept            : {stats['annotations_kept']}")
    print(f"  Lines skipped (###)         : {stats['lines_dont_care']}")
    print(f"  Lines skipped (charset)     : {stats['lines_charset_fail']}")
    print(f"  Lines skipped (bad bbox)    : {stats['lines_bad_bbox']}")
    print(f"  Lines skipped (bad parse)   : {stats['lines_bad_parse']}")
    print(f"\nWrote: {out_json}")
    if not args.no_copy:
        print(f"Images -> {out_imgs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
