"""
Filter ICDAR2019-style folders (--images, --gt) by script label.

- Drops image + GT when the image has no valid annotation whose script is in
  --allowed_scripts (same rules as convert_ICDAR2019: empty script -> "None",
  skip empty text and "###").
- For images that are kept, disallowed annotations are either removed
  (--mixed_policy remove) or kept with script column set to None
  (--mixed_policy none_script).

Images are copied to --out_images with format handling (GIF -> JPEG; other
formats saved as RGB in the original suffix when PIL supports it).
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".GIF",
    ".BMP",
    ".TIF",
    ".TIFF",
    ".WEBP",
]


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    default_icdar = repo / "Data" / "ICDAR2019"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images",
        type=str,
        default=str(default_icdar / "test_imgs"),
        help="Folder with ICDAR2019 images",
    )
    p.add_argument(
        "--gt",
        type=str,
        default=str(default_icdar / "test_gt"),
        help="Folder with per-image *.txt ground truth",
    )
    p.add_argument(
        "--allowed_scripts",
        type=str,
        required=True,
        help="Comma-separated script labels to treat as allowed, e.g. Latin,Arabic,Symbols,None",
    )
    p.add_argument(
        "--mixed_policy",
        type=str,
        choices=("remove", "none_script"),
        default="remove",
        help=(
            "If an image is kept but has both allowed and disallowed annotations: "
            "'remove' drops disallowed lines; 'none_script' keeps them with script set to None"
        ),
    )
    p.add_argument(
        "--out_images",
        type=str,
        default=None,
        help="Output image folder (default: <images_parent>/<images_name>_script_filtered)",
    )
    p.add_argument(
        "--out_gt",
        type=str,
        default=None,
        help="Output GT folder (default: <gt_parent>/<gt_name>_script_filtered)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions only; do not write files",
    )
    return p.parse_args()


def find_image(image_dir: Path, stem: str) -> tuple[Path, str] | None:
    for ext in IMAGE_EXTENSIONS:
        cand = image_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand, ext
    return None


def line_to_csv(parts: list[str]) -> str:
    """Serialize GT fields to one ICDAR-style CSV line."""
    row = parts[:8] + [parts[8], parts[9]]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(row)
    return buf.getvalue().rstrip("\n")


def parse_gt_line(line: str) -> tuple[list[str] | None, str | None]:
    line = line.strip()
    if not line:
        return None, "empty"
    try:
        parts = next(csv.reader([line]))
    except Exception:
        return None, "csv_error"
    if len(parts) < 10:
        return None, "short"
    return parts, None


def annotation_script_and_valid(parts: list[str]) -> tuple[str, bool]:
    """Script key (empty -> None) and whether this ann counts for retention (like convert_ICDAR2019)."""
    text = parts[9].strip() if len(parts) > 9 else ""
    if not text or text == "###":
        return "None", False
    script = parts[8].strip() if len(parts) > 8 else ""
    script_key = script if script else "None"
    return script_key, True


def convert_gif_to_rgb_jpg(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1])
            img = rgb_img
        elif img.mode != "RGB":
            img = img.convert("RGB")
        if getattr(img, "is_animated", False):
            img.seek(0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=95)


def save_image_rgb(src: Path, dst: Path) -> None:
    ext = dst.suffix.lower()
    with Image.open(src) as img:
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        if getattr(img, "is_animated", False):
            img.seek(0)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if ext in (".jpg", ".jpeg"):
            img.save(dst, "JPEG", quality=95)
        elif ext == ".png":
            img.save(dst, "PNG")
        elif ext == ".webp":
            img.save(dst, "WEBP", quality=95)
        elif ext in (".bmp",):
            img.save(dst, "BMP")
        elif ext in (".tif", ".tiff"):
            img.save(dst, "TIFF", compression="tiff_lzw")
        else:
            img.save(dst, "PNG")


def copy_or_convert_image(src: Path, out_images: Path, stem: str) -> str:
    """
    Write image under out_images; returns final file_name (may change .gif -> .jpg).
    """
    suf = src.suffix
    if suf.lower() == ".gif":
        name = f"{stem}.jpg"
        dst = out_images / name
        convert_gif_to_rgb_jpg(src, dst)
        return name
    name = src.name
    dst = out_images / name
    save_image_rgb(src, dst)
    return name


def process_gt_file(
    gt_path: Path,
    allowed: set[str],
    mixed_policy: str,
) -> tuple[str, list[str], dict]:
    """
    Returns (status, out_lines, stats).

    status:
      - 'drop_image' : no valid allowed-script annotation in file
      - 'keep'       : write out_lines (may be empty if something odd)
    """
    stats = {
        "raw_lines": 0,
        "kept": 0,
        "removed": 0,
        "relabeled_none": 0,
        "skipped_invalid": 0,
    }
    raw = gt_path.read_text(encoding="utf-8-sig").splitlines()
    parsed_rows: list[tuple[str, list[str] | None, str | None]] = []
    for line in raw:
        if not line.strip():
            continue
        stats["raw_lines"] += 1
        parts, err = parse_gt_line(line)
        parsed_rows.append((line, parts, err))

    has_allowed = False
    for _line, parts, err in parsed_rows:
        if err or parts is None:
            continue
        script_key, valid = annotation_script_and_valid(parts)
        if valid and script_key in allowed:
            has_allowed = True
            break

    if not has_allowed:
        return "drop_image", [], stats

    out_lines: list[str] = []
    for line, parts, err in parsed_rows:
        if err or parts is None:
            stats["skipped_invalid"] += 1
            out_lines.append(line.strip())
            continue
        script_key, valid = annotation_script_and_valid(parts)
        if not valid:
            out_lines.append(line.strip())
            stats["kept"] += 1
            continue

        if script_key in allowed:
            out_lines.append(line_to_csv(parts))
            stats["kept"] += 1
            continue

        if mixed_policy == "remove":
            stats["removed"] += 1
            continue

        parts = list(parts)
        parts[8] = "None"
        out_lines.append(line_to_csv(parts))
        stats["relabeled_none"] += 1
        stats["kept"] += 1

    return "keep", out_lines, stats


def main():
    args = parse_args()
    image_dir = Path(args.images)
    gt_dir = Path(args.gt)
    allowed = {s.strip() for s in args.allowed_scripts.split(",") if s.strip()}
    if not allowed:
        raise SystemExit("--allowed_scripts must contain at least one script label")

    if args.out_images:
        out_images = Path(args.out_images)
    else:
        out_images = image_dir.parent / f"{image_dir.name}_script_filtered"
    if args.out_gt:
        out_gt = Path(args.out_gt)
    else:
        out_gt = gt_dir.parent / f"{gt_dir.name}_script_filtered"

    if not image_dir.is_dir():
        raise SystemExit(f"Not a directory: {image_dir}")
    if not gt_dir.is_dir():
        raise SystemExit(f"Not a directory: {gt_dir}")

    print(f"Filtering by scripts: {', '.join(sorted(allowed))}")
    print(f"Policy: {args.mixed_policy}")
    print()
    
    summary = {
        "gt_files": 0,
        "images_dropped_no_allowed": 0,
        "images_missing_file": 0,
        "images_kept": 0,
        "lines_removed": 0,
        "lines_relabeled_none": 0,
        "lines_kept": 0,
        "invalid_gt_lines": 0,
    }

    if not args.dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)

    gt_files = list(gt_dir.glob("*.txt"))
    print(f"Processing {len(gt_files)} GT files...")
    
    for gt_path in sorted(gt_files):
        summary["gt_files"] += 1
        stem = gt_path.stem
        status, out_lines, st = process_gt_file(gt_path, allowed, args.mixed_policy)
        summary["lines_removed"] += st["removed"]
        summary["lines_relabeled_none"] += st["relabeled_none"]
        summary["lines_kept"] += st["kept"]
        summary["invalid_gt_lines"] += st["skipped_invalid"]

        if status == "drop_image":
            summary["images_dropped_no_allowed"] += 1
            if args.dry_run:
                print(f"  DROP: {stem}")
            continue

        found = find_image(image_dir, stem)
        if found is None:
            summary["images_missing_file"] += 1
            print(f"  WARN: {stem} - GT has allowed script but no image")
            continue

        src_img, _ext = found
        if args.dry_run:
            print(f"  KEEP: {stem}")
            summary["images_kept"] += 1
            continue

        final_name = copy_or_convert_image(src_img, out_images, stem)
        out_gt_path = out_gt / f"{stem}.txt"
        text = "\n".join(out_lines)
        if text and not text.endswith("\n"):
            text += "\n"
        out_gt_path.write_text(text, encoding="utf-8", newline="\n")
        summary["images_kept"] += 1
        
        # Show progress every 100 files
        if summary["images_kept"] % 100 == 0:
            print(f"  Processed {summary['images_kept']} kept images...")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Allowed scripts: {', '.join(sorted(allowed))}")
    print(f"Mixed policy   : {args.mixed_policy}")
    print()
    print(f"GT files scanned      : {summary['gt_files']}")
    print(f"Images dropped        : {summary['images_dropped_no_allowed']}")
    print(f"Images missing file   : {summary['images_missing_file']}")
    print(f"Images kept           : {summary['images_kept']}")
    print(f"GT lines kept         : {summary['lines_kept']}")
    print(f"GT lines removed      : {summary['lines_removed']}")
    print(f"GT lines -> None      : {summary['lines_relabeled_none']}")
    print(f"Invalid/unparsed lines: {summary['invalid_gt_lines']}")
    
    if args.dry_run:
        print("\n(DRY RUN - no files written)")
    else:
        print(f"\nOutput saved to:")
        print(f"  Images: {out_images}")
        print(f"  GT    : {out_gt}")
    print("=" * 60)


if __name__ == "__main__":
    main()