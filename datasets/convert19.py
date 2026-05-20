import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image


MAX_TEXT_LEN = 25


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    data_icdar = repo_root / "Data" / "ICDAR2019"
    parser = argparse.ArgumentParser("Convert ICDAR2019 to SPTSv2 JSON")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(data_icdar / "test_imgs"),
        help="Folder with ICDAR2019 images (train_imgs or test_imgs)",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=str(data_icdar / "test_gt"),
        help="Folder with per-image .txt ground truth",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(data_icdar / "test.json"),
        help="Output COCO-style JSON path",
    )
    parser.add_argument(
        "--charset_path",
        type=str,
        default=str(data_icdar / "icdar2019_charset.txt"),
        help="Single-line UTF-8 charset file",
    )
    parser.add_argument("--max_text_len", type=int, default=MAX_TEXT_LEN)
    parser.add_argument(
        "--allowed_scripts",
        type=str,
        default="Latin,Symbols,None",
        help=(
            "Comma-separated list of script labels to keep, e.g. 'Latin,Arabic,Symbols,None'. "
            "Leave empty (default) to keep ALL scripts. "
            "Images that have some annotations in discarded scripts keep their remaining "
            "allowed annotations; images whose every annotation is discarded are still "
            "written to the JSON with an empty annotation list (no images are deleted)."
        ),
    )
    return parser.parse_args()


def load_charset(charset_path):
    with open(Path(charset_path), "r", encoding="utf-8") as f:
        chars = f.read().rstrip("\n")
    if not chars:
        raise ValueError(f"Empty charset file: {charset_path}")
    return chars


def encode_text(text, char_to_idx, max_text_len, unk_idx, pad_idx):
    # Keep text multilingual as-is; do not normalize to ASCII.
    encoded = [char_to_idx.get(c, unk_idx) for c in text[:max_text_len]]
    if len(encoded) < max_text_len:
        encoded.extend([pad_idx] * (max_text_len - len(encoded)))
    return encoded

def quad_to_bbox_and_bezier(x1, y1, x2, y2, x3, y3, x4, y4):
    """Convert quadrilateral to bbox [x,y,w,h] and bezier_pts (16 values)"""
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    
    # SPTS v2 expects bbox as [x, y, width, height]
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
    
    # Create Bezier points (8 points * 2 coordinates = 16 values)
    bezier_pts = [
        x1, y1,  # point 1
        x2, y2,  # point 2
        x2, y2,  # point 3 (repeated)
        x3, y3,  # point 4
        x3, y3,  # point 5 (repeated)
        x4, y4,  # point 6
        x4, y4,  # point 7 (repeated)
        x1, y1   # point 8
    ]
    
    return bbox, bezier_pts

def convert_gif_to_jpg(gif_path):
    """Convert GIF to JPG, keep original GIF, return JPG path"""
    # Create JPG filename (replace .gif with .jpg)
    jpg_path = gif_path.with_suffix('.jpg')
    
    # Check if JPG already exists
    if jpg_path.exists():
        return jpg_path
    
    try:
        # Open GIF
        with Image.open(gif_path) as img:
            # Convert palette images with transparency to RGBA first
            if img.mode == 'P':
                img = img.convert('RGBA')
            
            # Convert to RGB (removes alpha channel for JPG)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                # Paste the image using alpha as mask if available
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # For GIFs, take the first frame
            if getattr(img, 'is_animated', False):
                img.seek(0)
            
            # Save as JPG with high quality
            img.save(jpg_path, 'JPEG', quality=95)
            
        print(f"  Converted: {gif_path.name} -> {jpg_path.name}")
        return jpg_path
    
    except Exception as e:
        print(f"  Error converting {gif_path.name}: {e}")
        return None

def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    output_json = Path(args.output_json)

    characters = load_charset(args.charset_path)
    char_to_idx = {ch: idx for idx, ch in enumerate(characters)}
    unk_idx = len(characters)
    pad_idx = len(characters) + 1

    # Build the allowed-script filter set (None = accept all).
    if args.allowed_scripts.strip():
        allowed_scripts = {s.strip() for s in args.allowed_scripts.split(",") if s.strip()}
        print(f"Script filter active — keeping: {sorted(allowed_scripts)}")
    else:
        allowed_scripts = None
        print("Script filter: OFF (all scripts kept)")

    diagnostics = {
        "gt_files_found": 0,
        "images_found": 0,
        "gif_converted": 0,
        "images_missing": [],
        "images_corrupted": [],
        "gt_files_empty": [],
        "total_annotations": 0,
        "valid_annotations": 0,
        "invalid_bbox": 0,
        "no_text": 0,
        "script_filtered": 0,      # annotations skipped because of script filter
        "parsing_errors": 0,
        "images_all_filtered": 0,  # images where every annotation was filtered out
        "per_script_kept": {},     # kept count per script label
        "per_script_dropped": {},  # dropped count per script label
    }

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(".txt")]
    diagnostics["gt_files_found"] = len(gt_files)
    print(f"Found {len(gt_files)} GT files")

    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".JPG", ".JPEG", ".PNG", ".GIF", ".bmp", ".tif", ".tiff"]
    valid_images = []

    for gt_file in gt_files:
        img_name_base = gt_file.replace(".txt", "")
        original_img_path = None
        original_ext = None
        for ext in image_extensions:
            candidate = image_dir / (img_name_base + ext)
            if candidate.exists():
                original_img_path = candidate
                original_ext = ext
                break

        if original_img_path is None:
            diagnostics["images_missing"].append(gt_file)
            continue

        if original_ext.lower() == ".gif":
            jpg_path = convert_gif_to_jpg(original_img_path)
            if jpg_path and jpg_path.exists():
                valid_images.append((gt_file, jpg_path.name, jpg_path))
                diagnostics["gif_converted"] += 1
            else:
                valid_images.append((gt_file, original_img_path.name, original_img_path))
        else:
            valid_images.append((gt_file, original_img_path.name, original_img_path))

    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for gt_file, img_name, img_path in valid_images:
        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size

            diagnostics["images_found"] += 1
            images.append({
                "id": image_id,
                "file_name": img_name,
                "width": w,
                "height": h,
            })

            gt_full_path = gt_dir / gt_file
            if not gt_full_path.exists() or gt_full_path.stat().st_size == 0:
                diagnostics["gt_files_empty"].append(gt_file)
                image_id += 1
                continue

            with open(gt_full_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    diagnostics["total_annotations"] += 1
                    try:
                        parts = next(csv.reader([line]))
                        if len(parts) < 10:
                            diagnostics["parsing_errors"] += 1
                            continue

                        x1, y1, x2, y2, x3, y3, x4, y4 = [float(v) for v in parts[:8]]
                        script = parts[8].strip() if len(parts) > 8 else ""
                        text = parts[9].strip() if len(parts) > 9 else ""

                        if not text or text == "###":
                            diagnostics["no_text"] += 1
                            continue

                        # --- Script filter ---
                        # An empty script label in the GT is treated as "None".
                        script_key = script if script else "None"
                        if allowed_scripts is not None and script_key not in allowed_scripts:
                            diagnostics["script_filtered"] += 1
                            diagnostics["per_script_dropped"][script_key] = (
                                diagnostics["per_script_dropped"].get(script_key, 0) + 1
                            )
                            continue
                        diagnostics["per_script_kept"][script_key] = (
                            diagnostics["per_script_kept"].get(script_key, 0) + 1
                        )

                        bbox, bezier_pts = quad_to_bbox_and_bezier(x1, y1, x2, y2, x3, y3, x4, y4)
                        if bbox[2] <= 0 or bbox[3] <= 0:
                            diagnostics["invalid_bbox"] += 1
                            continue

                        encoded_text = encode_text(
                            text=text,
                            char_to_idx=char_to_idx,
                            max_text_len=args.max_text_len,
                            unk_idx=unk_idx,
                            pad_idx=pad_idx,
                        )

                        annotations.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "rec": encoded_text,
                            "bezier_pts": bezier_pts,
                        })
                        diagnostics["valid_annotations"] += 1
                        ann_id += 1
                    except Exception:
                        diagnostics["parsing_errors"] += 1
                        continue

            # Track images where filtering left zero annotations.
            if allowed_scripts is not None:
                img_anns = [a for a in annotations if a["image_id"] == image_id]
                if len(img_anns) == 0:
                    diagnostics["images_all_filtered"] += 1

            image_id += 1
        except Exception as e:
            diagnostics["images_corrupted"].append((img_name, str(e)))
            continue

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text"}],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_format, f)

    print("\n=== Conversion done ===")
    print(f"Charset size       : {len(characters)}")
    print(f"Unknown token index: {unk_idx}")
    print(f"Pad token index    : {pad_idx}")
    print(f"Images in JSON     : {len(images)}  (no images removed)")
    print(f"Valid annotations  : {len(annotations)}")
    if allowed_scripts is not None:
        print(f"Script-filtered out: {diagnostics['script_filtered']} annotations")
        print(f"Images with 0 anns after filtering: {diagnostics['images_all_filtered']}")
        if diagnostics["per_script_kept"]:
            print("\nAnnotations KEPT per script:")
            for s, n in sorted(diagnostics["per_script_kept"].items(), key=lambda x: -x[1]):
                print(f"  {s:<20} {n}")
        if diagnostics["per_script_dropped"]:
            print("\nAnnotations DROPPED per script:")
            for s, n in sorted(diagnostics["per_script_dropped"].items(), key=lambda x: -x[1]):
                print(f"  {s:<20} {n}")
    print(f"\nSaved: {output_json}")


if __name__ == "__main__":
    main()