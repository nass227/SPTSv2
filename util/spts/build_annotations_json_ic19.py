    
"""

Build a COCO-style annotation JSON matching the ic13_test.json / ic15_test.json
spts schema used by this project.

Input: ICDAR2019-style annotation JSON
  {
    "gt_1234": [
      {"points": [[x1,y1],[x2,y2],...], "transcription": "text"},
      ...
    ],
    ...
  }
  Each top-level key is matched to an image file in --img-dir using the
  standard prefix/suffix strategies (e.g. "gt_1234" -> "img_1234.jpg").

Output JSON schema (matches ic13_test.json exactly):
  {
    "licenses": [],
    "info":     {},
    "categories": [{"id":1, "name":"text", "supercategory":"beverage",
                    "keypoints":["mean","xmin","x2","x3","xmax",
                                 "ymin","y2","y3","ymax","cross"]}],
    "images":   [{"coco_url","date_captured","file_name","flickr_url",
                  "id","license","width","height"}],
    "annotations": [{"id","image_id","category_id","bbox","area",
                     "iscrowd","bezier_pts","rec","rec_string"}]
  }

Usage:
  python build_annotations_json_ic19.py \\
      --json-input Data/ICDAR2019/train_labels.json \\
      --img-dir    Data/ICDAR2019/train_images \\
      -o           Data/ICDAR2019/train.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Character set — must match main_original.py / main.py --chars default.
#
# Index layout (95 chars):
#   0        : ' '  (space)
#   1  –  15 : !"#$%&'()*+,-./
#   16 –  25 : 0123456789
#   26 –  32 : :;<=>?@
#   33 –  58 : A-Z
#   59 –  64 : [\]^_`
#   65 –  90 : a-z
#   91 –  94 : {|}~
#
# PAD_IDX = 96   (index = ord(char) - 32)
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./'          # 0-16
    '0123456789'                   # 16-25
    ':;<=>?@'                      # 26-32
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 33-58
    '[\\]^_`'                      # 59-64
    'abcdefghijklmnopqrstuvwxyz'   # 65-90
    '{|}~'                         # 91-94  → total 95 chars
)

MAX_LEN: int = 25
PAD_IDX: int = 96   # index 95 is reserved for OOV/unknown

# ---------------------------------------------------------------------------
# Image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".TIFF"]

# COCO category block matching ic13/ic15 spts JSON
CATEGORIES = [
    {
        "id": 1,
        "name": "text",
        "supercategory": "beverage",
        "keypoints": ["mean", "xmin", "x2", "x3", "xmax",
                      "ymin", "y2", "y3", "ymax", "cross"],
    }
]


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
def encode_text(text: str, chars: str = CHARS, max_len: int = MAX_LEN,
                pad_idx: int = PAD_IDX) -> list[int]:
    """Encode text to a fixed-length integer list (charset indices).

    Characters not in *chars* are silently skipped.
    Padded with *pad_idx* to exactly *max_len* elements.
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
    """Return 16 floats (two cubic bezier rails) from the four quad corners.

    Top edge  L->R : P0=top-left,   P1=1/3, P2=2/3, P3=top-right
    Bottom edge R->L : P4=bot-right, P5=1/3, P6=2/3, P7=bot-left
    """
    t_cx1 = x1 + (x2 - x1) / 3;  t_cy1 = y1 + (y2 - y1) / 3
    t_cx2 = x1 + 2*(x2 - x1)/3;  t_cy2 = y1 + 2*(y2 - y1)/3
    b_cx1 = x3 + (x4 - x3) / 3;  b_cy1 = y3 + (y4 - y3) / 3
    b_cx2 = x3 + 2*(x4 - x3)/3;  b_cy2 = y3 + 2*(y4 - y3)/3
    return [
        x1,    y1,    t_cx1, t_cy1, t_cx2, t_cy2, x2,    y2,
        x3,    y3,    b_cx1, b_cy1, b_cx2, b_cy2, x4,    y4,
    ]


def _order_quad_corners(
    points: list,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Return (tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y) from a point list.

    For exactly 4 points: splits into top-2 / bottom-2 by y, then by x.
    For other counts: falls back to AABB corners.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]

    if len(pts) == 4:
        by_y = sorted(pts, key=lambda p: p[1])
        top2 = sorted(by_y[:2], key=lambda p: p[0])
        bot2 = sorted(by_y[2:], key=lambda p: p[0])
        tl, tr = top2[0], top2[1]
        bl, br = bot2[0], bot2[1]
    else:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
        tl = [xmin, ymin]; tr = [xmax, ymin]
        br = [xmax, ymax]; bl = [xmin, ymax]

    return (tl[0], tl[1],  tr[0], tr[1],
            br[0], br[1],  bl[0], bl[1])


# ---------------------------------------------------------------------------
# Image matching
# ---------------------------------------------------------------------------
def _find_exact(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def find_image(img_dir: Path, key: str) -> Optional[Path]:
    """Try multiple prefix/suffix transformations of *key* to find its image."""
    result = _find_exact(img_dir, key)
    if result:
        return result

    if key.startswith("gt_"):
        inner = key[3:]
        for prefix in ("", "img_", "image_"):
            result = _find_exact(img_dir, prefix + inner)
            if result:
                return result

    if not key.startswith("gt_"):
        result = _find_exact(img_dir, "gt_" + key)
        if result:
            return result

    for pre in ("", "img_", "image_", "gt_", "gt_img_"):
        for suf in ("", "_img", "_image"):
            result = _find_exact(img_dir, pre + key + suf)
            if result:
                return result

    for suf in ("_gt", "_img", "_image"):
        if key.endswith(suf):
            result = find_image(img_dir, key[: -len(suf)])
            if result:
                return result

    m = re.match(r"^\d+_(.+)$", key)
    if m:
        result = find_image(img_dir, m.group(1))
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _image_entry(image_id: int, file_name: str, width: int, height: int) -> dict:
    return {
        "coco_url":      "",
        "date_captured": "",
        "file_name":     file_name,
        "flickr_url":    "",
        "id":            image_id,
        "license":       0,
        "width":         width,
        "height":        height,
    }


def _coco_root(images: list, annotations: list) -> dict:
    return {
        "licenses":    [],
        "info":        {},
        "categories":  list(CATEGORIES),
        "images":      images,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_json(
    json_path: Path,
    img_dir: Path,
    *,
    chars: str = CHARS,
    max_len: int = MAX_LEN,
    pad_idx: int = PAD_IDX,
) -> tuple[dict, dict]:
    """Build a spts-format COCO JSON from an ICDAR2019-style annotation JSON.

    Input JSON shape::

        {
          "gt_1234": [
            {"points": [[x1,y1],[x2,y2],...], "transcription": "text"},
            ...
          ],
          ...
        }

    Returns ``(coco_dict, stats)``.
    Images whose every annotation fails validation are excluded entirely.
    """
    with json_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at root, got {type(data).__name__}"
        )

    images_out: list[dict] = []
    annotations_out: list[dict] = []
    image_id = 1
    ann_id = 1

    stats: dict = {
        "keys_total":               len(data),
        "images_matched":           0,
        "images_missing":           0,
        "images_no_ann":            0,
        "annotations":              0,
        "skipped_no_transcription": 0,
        "skipped_bad_points":       0,
        "skipped_zero_area":        0,
    }

    print(f"[INFO] JSON keys (logical images): {stats['keys_total']}")

    for key, ann_list in data.items():
        img_path = find_image(img_dir, key)
        if img_path is None:
            stats["images_missing"] += 1
            continue

        try:
            with PILImage.open(img_path) as im:
                width, height = im.width, im.height
        except Exception as exc:
            print(f"  [WARN] Cannot open {img_path}: {exc}", file=sys.stderr)
            stats["images_missing"] += 1
            continue

        if not isinstance(ann_list, list):
            stats["images_no_ann"] += 1
            continue

        valid_anns: list[dict] = []

        for ann in ann_list:
            if not isinstance(ann, dict):
                continue

            transcription = (ann.get("transcription")
                             or ann.get("text")
                             or ann.get("label"))
            if not transcription:
                stats["skipped_no_transcription"] += 1
                continue

            points = ann.get("points") or ann.get("bbox_points") or []
            if not points:
                stats["skipped_bad_points"] += 1
                continue

            # Normalise flat [x1,y1,x2,y2,...] to [[x,y],...]
            if points and not isinstance(points[0], (list, tuple)):
                if len(points) >= 4 and len(points) % 2 == 0:
                    points = [[points[i], points[i+1]]
                              for i in range(0, len(points), 2)]
                else:
                    stats["skipped_bad_points"] += 1
                    continue

            if len(points) < 2:
                stats["skipped_bad_points"] += 1
                continue

            # Determine quad corners
            if len(points) >= 4:
                qx1, qy1, qx2, qy2, qx3, qy3, qx4, qy4 = _order_quad_corners(points)
            else:
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                qx1, qy1 = min(xs), min(ys)
                qx2, qy2 = max(xs), min(ys)
                qx3, qy3 = max(xs), max(ys)
                qx4, qy4 = min(xs), max(ys)

            # Clamp to image bounds
            def cx(v: float) -> float: return max(0.0, min(v, float(width)))
            def cy(v: float) -> float: return max(0.0, min(v, float(height)))
            qx1, qy1 = cx(qx1), cy(qy1)
            qx2, qy2 = cx(qx2), cy(qy2)
            qx3, qy3 = cx(qx3), cy(qy3)
            qx4, qy4 = cx(qx4), cy(qy4)

            xs2 = [qx1, qx2, qx3, qx4]; ys2 = [qy1, qy2, qy3, qy4]
            xmin, xmax = min(xs2), max(xs2)
            ymin, ymax = min(ys2), max(ys2)
            bw, bh = xmax - xmin, ymax - ymin
            if bw <= 0 or bh <= 0:
                stats["skipped_zero_area"] += 1
                continue

            valid_anns.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": 1,
                "bbox":        [xmin, ymin, bw, bh],
                "area":        float(bw * bh),
                "iscrowd":     0,
                "bezier_pts":  bezier_from_quad(
                                   qx1, qy1, qx2, qy2, qx3, qy3, qx4, qy4),
                "rec":         encode_text(str(transcription), chars, max_len, pad_idx),
                "rec_string":  str(transcription),
            })
            ann_id += 1
            stats["annotations"] += 1

        if not valid_anns:
            stats["images_no_ann"] += 1
            continue

        images_out.append(_image_entry(image_id, img_path.name, width, height))
        annotations_out.extend(valid_anns)
        stats["images_matched"] += 1
        image_id += 1

    return _coco_root(images_out, annotations_out), stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an spts-format COCO JSON from an ICDAR2019-style "
            "annotation JSON (gt_key -> [{points, transcription}])."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--json-input", required=True, metavar="PATH",
                   help="ICDAR2019-style annotation JSON.")
    p.add_argument("--img-dir", required=True, metavar="PATH",
                   help="Folder containing the image files.")
    p.add_argument("--output", "-o", required=True, metavar="PATH",
                   help="Output .json path.")
    p.add_argument("--max-len", type=int, default=MAX_LEN,
                   help=f"Max text length before truncation (default {MAX_LEN}).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    json_path = Path(args.json_input).resolve()
    img_dir   = Path(args.img_dir).resolve()
    out       = Path(args.output).resolve()

    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 2
    if not img_dir.is_dir():
        print(f"Error: image directory not found: {img_dir}", file=sys.stderr)
        return 2

    print(f"JSON    : {json_path}")
    print(f"Img dir : {img_dir}")
    print(f"Output  : {out}")
    print(f"Charset : {len(CHARS)} chars  |  max_len={args.max_len}  pad_idx={PAD_IDX}")
    print()

    data, stats = build_json(
        json_path, img_dir,
        chars=CHARS,
        max_len=args.max_len,
        pad_idx=PAD_IDX,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print()
    print("=" * 50)
    print("Build summary")
    print("=" * 50)
    print(f"  JSON keys (logical images)  : {stats['keys_total']}")
    print(f"  Images matched              : {stats['images_matched']}")
    print(f"  Images not found on disk    : {stats['images_missing']}")
    print(f"  Images with no valid ann    : {stats['images_no_ann']}")
    print(f"  Annotations written         : {stats['annotations']}")
    skipped = (stats["skipped_no_transcription"]
               + stats["skipped_bad_points"]
               + stats["skipped_zero_area"])
    if skipped:
        print(f"  Skipped annotations         : {skipped}"
              f"  (no text: {stats['skipped_no_transcription']},"
              f" bad points: {stats['skipped_bad_points']},"
              f" zero area: {stats['skipped_zero_area']})")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
