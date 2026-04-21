"""
Build a COCO-style annotation JSON.  Two input modes:

  --gt-dir   path/to/gt_txt_folder   (one .txt per image, any supported format)
  --json-input path/to/labels.json   (ICDAR2019 / per-key JSON with points + transcription)

All annotations are treated as clean and valid; no filtering is applied.

Supported GT line formats (--gt-dir mode):
  1. x1,y1,x2,y2,text              (AABB, 4 coords)
  2. x1,y1,x2,y2,x3,y3,x4,y4,text (quad, 8 coords, no script)
  3. x1,y1,...,x4,y4,SCRIPT,text   (quad + script label, 10+ fields)
  4. Flexible separators: comma / space / tab / semicolon / pipe
  5. Quoted text: "Royal" or 'London'

JSON input format (--json-input mode):
  {
    "gt_1234": [
      {"points": [[x1,y1],[x2,y2],...], "transcription": "text"},
      ...
    ],
    ...
  }
  Keys are matched to image files using the same prefix strategies as --gt-dir.

Output JSON schema (both modes):
  {
    "images":      [{"id", "file_name", "width", "height"}, ...],
    "annotations": [{"id", "image_id", "category_id",
                      "bbox", "area", "iscrowd",
                      "bezier_pts", "rec"[, "rec_string"]}, ...],
    "categories":  [{"id": 1, "name": "text"}]
  }

Usage:
  # From GT .txt folder
  python build_annotations_json.py --gt-dir PATH --img-dir PATH -o out.json

  # From ICDAR-style JSON
  python build_annotations_json.py --json-input PATH --img-dir PATH -o out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Character set  (128 characters — must match training config exactly)
#
# Index layout:
#   0   –  93 : printable ASCII  ! " # $ % & ' ( ) * + , - . / 0-9 : ; < = >
#                                ? @ A-Z [ \ ] ^ _ ` a-z { | } ~
#   94  – 111 : French lowercase  à â ä é è ê ë î ï ô ù û ü ÿ æ œ ç  (18 chars)
#   112 – 129 : French uppercase  À Â Ä É È Ê Ë Î Ï Ô Ù Û Ü Ÿ Æ Œ Ç  (18 chars)
#
# PAD_IDX = 
# Characters not in CHARS are silently dropped during encoding.
# ---------------------------------------------------------------------------
CHARS: str = (
    # indices 0-93: printable ASCII 33-126
    '!"#$%&\'()*+,-./'          # 0-15
    '0123456789'                 # 16-25
    ':;<=>?@'                    # 26-32
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # 33-58
    '[\\]^_`'                    # 59-64
    'abcdefghijklmnopqrstuvwxyz' # 65-90
    '{|}~'                       # 91-93 → total 94 chars (indices 0-93)
    # indices 94-111: French lowercase (18 chars)
    'àâäéèêëîïôùûüÿæœç'
    # indices 112-129: French uppercase (18 chars)
    'ÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ'
)

MAX_LEN: int = 25
PAD_IDX: int = 96   # padding token

# ---------------------------------------------------------------------------
# Image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".TIFF"]

# Separators accepted between numbers in GT lines
_NUM_SEPS = frozenset(" \t,;|")


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
def encode_text(text: str, chars: str = CHARS, max_len: int = MAX_LEN,
                pad_idx: int = PAD_IDX) -> list[int]:
    """Encode text to a fixed-length integer list.

    Characters not in `chars` are silently dropped.
    The list is padded with `pad_idx` to exactly `max_len` elements.
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
# Bézier control points from AABB
# ---------------------------------------------------------------------------
def bezier_from_aabb(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Return exactly 16 floats representing the two horizontal rails of a text box.

    Top edge (left → right):    (x1,y1) (x_a,y1) (x_b,y1) (x2,y1)
    Bottom edge (right → left): (x2,y2) (x_b,y2) (x_a,y2) (x1,y2)
    """
    x_a = x1 + (x2 - x1) / 3
    x_b = x1 + 2 * (x2 - x1) / 3
    return [
        x1, y1,  x_a, y1,  x_b, y1,  x2, y1,   # top edge — left to right
        x2, y2,  x_b, y2,  x_a, y2,  x1, y2,   # bottom edge — right to left
    ]


# ---------------------------------------------------------------------------
# GT line parsing
# ---------------------------------------------------------------------------
def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _parse_n_numbers(fields: list[str], n: int) -> Optional[list[float]]:
    if len(fields) < n:
        return None
    nums: list[float] = []
    for f in fields[:n]:
        try:
            nums.append(float(f.strip()))
        except ValueError:
            return None
    return nums


def _read_number_flexible(line: str, pos: int) -> tuple[Optional[float], int]:
    n = len(line)
    while pos < n and line[pos] in _NUM_SEPS:
        pos += 1
    if pos >= n:
        return None, pos
    start = pos
    if line[pos] in "+-":
        pos += 1
    saw_digit = False
    dot_seen = False
    while pos < n:
        ch = line[pos]
        if ch.isdigit():
            saw_digit = True
            pos += 1
        elif ch == "." and not dot_seen:
            dot_seen = True
            pos += 1
        else:
            break
    if not saw_digit:
        return None, start
    try:
        v = float(line[start:pos])
    except ValueError:
        return None, start
    return v, pos


def _parse_four_flexible(line: str) -> tuple[Optional[list[float]], str]:
    """Read exactly 4 leading numbers with any separator, return (nums, label)."""
    nums: list[float] = []
    pos = 0
    for _ in range(4):
        val, pos = _read_number_flexible(line, pos)
        if val is None:
            return None, ""
        nums.append(val)
    while pos < len(line) and line[pos] in _NUM_SEPS:
        pos += 1
    label = _strip_quotes(line[pos:])
    return nums, label


def _nums_to_aabb(nums: list[float]) -> tuple[float, float, float, float]:
    """Convert 4- or 8-number coordinate list to (x1, y1, x2, y2)."""
    if len(nums) == 4:
        xa, xb = min(nums[0], nums[2]), max(nums[0], nums[2])
        ya, yb = min(nums[1], nums[3]), max(nums[1], nums[3])
        return xa, ya, xb, yb
    xs = nums[0::2]
    ys = nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def parse_gt_line(raw: str) -> Optional[tuple[float, float, float, float, str]]:
    """Parse one GT line.  Returns (x1, y1, x2, y2, text) or None if unparseable."""
    line = raw.strip()
    if not line:
        return None

    try:
        fields = list(next(csv.reader([line])))
    except Exception:
        fields = [line]

    fields = [_strip_quotes(f.strip()) for f in fields]

    # 8-coord quad (± script label)
    if len(fields) >= 9:
        nums = _parse_n_numbers(fields, 8)
        if nums is not None:
            tail = fields[8:]
            # detect and skip a script-label field (e.g. "Latin", "Arabic")
            if (len(tail) >= 2
                    and re.match(r'^[A-Za-z][A-Za-z\-_]{0,19}$', tail[0])
                    and not tail[0].isdigit()):
                text = ",".join(tail[1:]).strip()
            else:
                text = ",".join(tail).strip()
            text = _strip_quotes(text)
            if not text:
                return None
            return *_nums_to_aabb(nums), text

    # 4-coord AABB (CSV)
    if len(fields) >= 5:
        nums = _parse_n_numbers(fields, 4)
        if nums is not None:
            text = _strip_quotes(",".join(fields[4:]).strip())
            if not text:
                return None
            return *_nums_to_aabb(nums), text

    # Flexible separators (spaces / mixed)
    nums4, label = _parse_four_flexible(line)
    if nums4 is not None and label:
        return *_nums_to_aabb(nums4), label

    return None


# ---------------------------------------------------------------------------
# Image matching
# ---------------------------------------------------------------------------
def _find_exact(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def find_image(img_dir: Path, gt_stem: str) -> Optional[Path]:
    """Try multiple prefix/suffix strategies to pair a GT stem with its image."""
    result = _find_exact(img_dir, gt_stem)
    if result:
        return result

    if gt_stem.startswith("gt_"):
        inner = gt_stem[3:]
        for prefix in ("", "img_", "image_"):
            result = _find_exact(img_dir, prefix + inner)
            if result:
                return result

    if not gt_stem.startswith("gt_"):
        result = _find_exact(img_dir, "gt_" + gt_stem)
        if result:
            return result

    for pre in ("", "img_", "image_", "gt_", "gt_img_"):
        for suf in ("", "_img", "_image"):
            result = _find_exact(img_dir, pre + gt_stem + suf)
            if result:
                return result

    for suf in ("_gt", "_img", "_image"):
        if gt_stem.endswith(suf):
            result = find_image(img_dir, gt_stem[: -len(suf)])
            if result:
                return result

    m = re.match(r"^\d+_(.+)$", gt_stem)
    if m:
        result = find_image(img_dir, m.group(1))
        if result:
            return result

    return None


def get_image_size(img_path: Path) -> tuple[int, int]:
    with PILImage.open(img_path) as im:
        return im.width, im.height


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------
def build_json(
    gt_dir: Path,
    img_dir: Path,
    *,
    chars: str = CHARS,
    max_len: int = MAX_LEN,
    pad_idx: int = PAD_IDX,
    include_rec_string: bool = False,
) -> tuple[dict, dict, list[str]]:
    images_out: list[dict] = []
    annotations_out: list[dict] = []
    image_id = 1
    ann_id = 1

    gt_files = sorted(
        p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"
    )
    print(f"\n[INFO] Found {len(gt_files)} GT files in {gt_dir}")

    unmatched: list[str] = []
    stats = {
        "gt_files": len(gt_files),
        "matched_images": 0,
        "annotations": 0,
        "skipped_lines": 0,   # empty lines or lines with bad / missing coordinates
        "unmatched_gt": 0,
    }

    for gt_path in gt_files:
        img_path = find_image(img_dir, gt_path.stem)
        if img_path is None:
            unmatched.append(gt_path.name)
            stats["unmatched_gt"] += 1
            continue

        try:
            width, height = get_image_size(img_path)
        except Exception as e:
            print(f"  [WARN] cannot open image {img_path}: {e}", file=sys.stderr)
            continue

        images_out.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        stats["matched_images"] += 1

        try:
            raw = gt_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = gt_path.read_text(encoding="utf-8", errors="replace")

        for line in raw.splitlines():
            parsed = parse_gt_line(line)
            if parsed is None:
                stats["skipped_lines"] += 1
                continue

            x1, y1, x2, y2, text = parsed

            # clamp to image bounds
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(float(width), x2);  y2 = min(float(height), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stats["skipped_lines"] += 1
                continue

            bbox = [x1, y1, bw, bh]
            bezier_pts = bezier_from_aabb(x1, y1, x2, y2)
            rec = encode_text(text, chars, max_len, pad_idx)

            ann: dict = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bw * bh,
                "iscrowd": 0,
                "bezier_pts": bezier_pts,
                "rec": rec,
            }
            if include_rec_string:
                ann["rec_string"] = text

            annotations_out.append(ann)
            ann_id += 1
            stats["annotations"] += 1

        image_id += 1

    print(f"[INFO] Total images processed : {len(images_out)}")
    print(f"[INFO] Total annotations      : {len(annotations_out)}")

    result = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": [{"id": 1, "name": "text"}],
    }
    return result, stats, unmatched


# ---------------------------------------------------------------------------
# JSON-input builder  (ICDAR2019 / "gt_key": [{points, transcription}] format)
# ---------------------------------------------------------------------------
def build_json_from_json_input(
    json_path: Path,
    img_dir: Path,
    *,
    chars: str = CHARS,
    max_len: int = MAX_LEN,
    pad_idx: int = PAD_IDX,
    include_rec_string: bool = False,
) -> tuple[dict, dict]:
    """Build COCO JSON from a per-key annotation JSON (ICDAR2019 style).

    Expected JSON shape:
      {
        "gt_1234": [
          {"points": [[x1,y1],[x2,y2],...], "transcription": "text"},
          ...
        ],
        ...
      }
    Each top-level key is matched to an image file using the same
    prefix/suffix strategies as the GT-folder mode.
    """
    with json_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at root, got {type(data).__name__}")

    images_out: list[dict] = []
    annotations_out: list[dict] = []
    image_id = 1
    ann_id = 1

    stats = {
        "keys_total": len(data),
        "images_matched": 0,
        "images_missing": 0,
        "annotations": 0,
        "skipped_no_transcription": 0,
        "skipped_bad_points": 0,
        "skipped_zero_area": 0,
    }

    print(f"\n[INFO] JSON keys (logical images): {stats['keys_total']}")

    for key, ann_list in data.items():
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

            # Accept both [[x,y],...] and flat [x1,y1,x2,y2,...] coordinate formats
            points = ann.get("points") or ann.get("bbox_points") or []
            if not points:
                stats["skipped_bad_points"] += 1
                continue

            # Normalise to [[x,y], ...] if given as flat list
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

            # Clamp to image bounds
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(float(width), x2);  y2 = min(float(height), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stats["skipped_zero_area"] += 1
                continue

            text = str(transcription)
            bbox = [x1, y1, bw, bh]
            bezier_pts = bezier_from_aabb(x1, y1, x2, y2)
            rec = encode_text(text, chars, max_len, pad_idx)

            ann_out: dict = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bw * bh,
                "iscrowd": 0,
                "bezier_pts": bezier_pts,
                "rec": rec,
            }
            if include_rec_string:
                ann_out["rec_string"] = text

            annotations_out.append(ann_out)
            ann_id += 1
            stats["annotations"] += 1

        image_id += 1

    print(f"[INFO] Images matched  : {stats['images_matched']}")
    print(f"[INFO] Images missing  : {stats['images_missing']}")
    print(f"[INFO] Annotations     : {stats['annotations']}")
    skipped = (stats["skipped_no_transcription"]
               + stats["skipped_bad_points"]
               + stats["skipped_zero_area"])
    if skipped:
        print(f"[INFO] Skipped entries : {skipped}"
              f"  (no text: {stats['skipped_no_transcription']},"
              f" bad points: {stats['skipped_bad_points']},"
              f" zero-area: {stats['skipped_zero_area']})")

    return {
        "images": images_out,
        "annotations": annotations_out,
        "categories": [{"id": 1, "name": "text"}],
    }, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a COCO-style JSON from GT files or an ICDAR-style JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
input modes (mutually exclusive, one is required):
  --gt-dir PATH      folder of .txt GT files (one per image)
  --json-input PATH  ICDAR-style JSON: {"gt_key": [{points, transcription}]}
""",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt-dir",     metavar="PATH", help="Folder of .txt GT files.")
    src.add_argument("--json-input", metavar="PATH", help="ICDAR-style annotation JSON.")

    p.add_argument("--img-dir", required=True, metavar="PATH",
                   help="Folder containing the image files.")
    p.add_argument("--output", "-o", required=True, metavar="PATH",
                   help="Output .json path.")
    p.add_argument("--max-len", type=int, default=MAX_LEN,
                   help=f"Max text length before truncation (default {MAX_LEN}).")
    p.add_argument("--pad-idx", type=int, default=PAD_IDX,
                   help=f"Rec padding index (default {PAD_IDX}).")
    p.add_argument("--rec-string", action="store_true",
                   help="Include original text as rec_string field on each annotation.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    img_dir = Path(args.img_dir).resolve()
    out     = Path(args.output).resolve()

    if not img_dir.is_dir():
        print(f"Error: image directory not found: {img_dir}", file=sys.stderr)
        return 2

    print(f"Img dir : {img_dir}")
    print(f"Output  : {out}")
    print(f"Charset : {len(CHARS)} chars  |  max_len={args.max_len}  pad_idx={args.pad_idx}")
    print(f"Charset : {CHARS!r}")
    print()

    common = dict(
        chars=CHARS,
        max_len=args.max_len,
        pad_idx=args.pad_idx,
        include_rec_string=args.rec_string,
    )

    if args.json_input:
        json_path = Path(args.json_input).resolve()
        if not json_path.is_file():
            print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
            return 2
        print(f"Mode    : JSON input")
        print(f"JSON    : {json_path}")
        data, stats = build_json_from_json_input(json_path, img_dir, **common)
        summary_lines = [
            f"  JSON keys (logical images) : {stats['keys_total']}",
            f"  Images matched             : {stats['images_matched']}",
            f"  Images missing             : {stats['images_missing']}",
            f"  Annotations written        : {stats['annotations']}",
        ]

    else:
        gt_dir = Path(args.gt_dir).resolve()
        if not gt_dir.is_dir():
            print(f"Error: GT directory not found: {gt_dir}", file=sys.stderr)
            return 2
        print(f"Mode    : GT folder")
        print(f"GT dir  : {gt_dir}")
        data, stats, unmatched = build_json(gt_dir, img_dir, **common)
        summary_lines = [
            f"  GT files found      : {stats['gt_files']}",
            f"  Images matched      : {stats['matched_images']}",
            f"  Unmatched GT files  : {stats['unmatched_gt']}",
            f"  Annotations written : {stats['annotations']}",
            f"  Lines skipped       : {stats['skipped_lines']}  (empty / bad coords)",
        ]
        if unmatched:
            summary_lines.append(f"\n  Unmatched GT files (first 20):")
            for name in unmatched[:20]:
                summary_lines.append(f"    {name}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("\n=== Build summary ===")
    for line in summary_lines:
        print(line)
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
