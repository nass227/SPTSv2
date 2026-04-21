# """
# Filter ICDAR2019-style folders (--images, --gt) by script label.

# - Drops image + GT when the image has no valid annotation whose script is in
#   --allowed_scripts (same rules as convert_ICDAR2019: empty script -> "None",
#   skip empty text and "###").
# - For images that are kept, disallowed annotations are either removed
#   (--mixed_policy remove) or kept with script column set to None
#   (--mixed_policy none_script).

# Images are copied to --out_images with format handling (GIF -> JPEG; other
# formats saved as RGB in the original suffix when PIL supports it).
# """

# from __future__ import annotations

# import argparse
# import csv
# from pathlib import Path

# from PIL import Image

# IMAGE_EXTENSIONS = [
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".gif",
#     ".bmp",
#     ".tif",
#     ".tiff",
#     ".webp",
#     ".JPG",
#     ".JPEG",
#     ".PNG",
#     ".GIF",
#     ".BMP",
#     ".TIF",
#     ".TIFF",
#     ".WEBP",
# ]


# def parse_args():
#     repo = Path(__file__).resolve().parent.parent
#     default_icdar = repo / "Data" / "ICDAR2019"
#     p = argparse.ArgumentParser(description=__doc__)
#     p.add_argument(
#         "--images",
#         type=str,
#         default=str(default_icdar / "test_imgs"),
#         help="Folder with ICDAR2019 images",
#     )
#     p.add_argument(
#         "--gt",
#         type=str,
#         default=str(default_icdar / "test_gt"),
#         help="Folder with per-image *.txt ground truth",
#     )
#     p.add_argument(
#         "--allowed_scripts",
#         type=str,
#         required=True,
#         help="Comma-separated script labels to treat as allowed, e.g. Latin,Arabic,Symbols,None",
#     )
#     p.add_argument(
#         "--mixed_policy",
#         type=str,
#         choices=("remove", "none_script"),
#         default="remove",
#         help=(
#             "If an image is kept but has both allowed and disallowed annotations: "
#             "'remove' drops disallowed lines; 'none_script' keeps them with script set to None"
#         ),
#     )
#     p.add_argument(
#         "--out_images",
#         type=str,
#         default=None,
#         help="Output image folder (default: <images_parent>/<images_name>_script_filtered)",
#     )
#     p.add_argument(
#         "--out_gt",
#         type=str,
#         default=None,
#         help="Output GT folder (default: <gt_parent>/<gt_name>_script_filtered)",
#     )
#     p.add_argument(
#         "--dry_run",
#         action="store_true",
#         help="Print actions only; do not write files",
#     )
#     return p.parse_args()


# def find_image(image_dir: Path, stem: str) -> tuple[Path, str] | None:
#     for ext in IMAGE_EXTENSIONS:
#         cand = image_dir / f"{stem}{ext}"
#         if cand.is_file():
#             return cand, ext
#     return None
# def find_image_with_prefix(image_dir: Path, gt_stem: str) -> tuple[Path, str] | None:
#     """
#     Find image file with possible prefix variations.
#     Handles cases like:
#     - GT: gt_img_123 -> Image: img_123
#     - GT: img_123 -> Image: img_123
#     - GT: gt_123 -> Image: 123 or gt_123
#     - GT: gt_123 -> Image: gt_123 (same name)
#     - GT: 123 -> Image: gt_123
#     """
#     # Try exact match first (most common case when names match exactly)
#     result = find_image(image_dir, gt_stem)
#     if result:
#         return result
    
#     # Try removing common prefixes from GT stem
#     prefixes_to_try = ["gt_", "gt_img_", "img_", "image_"]
#     for prefix in prefixes_to_try:
#         if gt_stem.startswith(prefix):
#             # Remove prefix and try
#             stripped = gt_stem[len(prefix):]
#             result = find_image(image_dir, stripped)
#             if result:
#                 return result
    
#     # Try adding common prefixes (for cases where GT has no prefix but image does)
#     prefixes_to_add = ["gt_", "img_", "image_"]
#     for prefix in prefixes_to_add:
#         result = find_image(image_dir, f"{prefix}{gt_stem}")
#         if result:
#             return result
    
#     # Special case: try removing 'gt_' and adding 'img_' 
#     # (e.g., gt_123 -> img_123)
#     if gt_stem.startswith("gt_"):
#         stripped = gt_stem[3:]  # Remove 'gt_'
#         result = find_image(image_dir, f"img_{stripped}")
#         if result:
#             return result
    
#     return None
 

# def line_to_csv(parts: list[str]) -> str:
#     """Serialize GT fields to one ICDAR-style CSV line."""
#     row = parts[:8] + [parts[8], parts[9]]
#     buf = []
#     w = csv.writer(buf, lineterminator="")
#     w.writerow(row)
#     return buf[0]


# def parse_gt_line(line: str) -> tuple[list[str] | None, str | None]:
#     line = line.strip()
#     if not line:
#         return None, "empty"
#     try:
#         parts = next(csv.reader([line]))
#     except Exception:
#         return None, "csv_error"
#     if len(parts) < 10:
#         return None, "short"
#     return parts, None


# def annotation_script_and_valid(parts: list[str]) -> tuple[str, bool]:
#     """Script key (empty -> None) and whether this ann counts for retention (like convert_ICDAR2019)."""
#     text = parts[9].strip() if len(parts) > 9 else ""
#     if not text or text == "###":
#         return "None", False
#     script = parts[8].strip() if len(parts) > 8 else ""
#     script_key = script if script else "None"
#     return script_key, True


# def convert_gif_to_rgb_jpg(src: Path, dst: Path) -> None:
#     with Image.open(src) as img:
#         if img.mode == "P":
#             img = img.convert("RGBA")
#         if img.mode in ("RGBA", "LA"):
#             rgb_img = Image.new("RGB", img.size, (255, 255, 255))
#             rgb_img.paste(img, mask=img.split()[-1])
#             img = rgb_img
#         elif img.mode != "RGB":
#             img = img.convert("RGB")
#         if getattr(img, "is_animated", False):
#             img.seek(0)
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         img.save(dst, "JPEG", quality=95)


# def save_image_rgb(src: Path, dst: Path) -> None:
#     ext = dst.suffix.lower()
#     with Image.open(src) as img:
#         if img.mode == "P":
#             img = img.convert("RGBA")
#         if img.mode in ("RGBA", "LA"):
#             bg = Image.new("RGB", img.size, (255, 255, 255))
#             bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
#             img = bg
#         elif img.mode != "RGB":
#             img = img.convert("RGB")
#         if getattr(img, "is_animated", False):
#             img.seek(0)
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         if ext in (".jpg", ".jpeg"):
#             img.save(dst, "JPEG", quality=95)
#         elif ext == ".png":
#             img.save(dst, "PNG")
#         elif ext == ".webp":
#             img.save(dst, "WEBP", quality=95)
#         elif ext in (".bmp",):
#             img.save(dst, "BMP")
#         elif ext in (".tif", ".tiff"):
#             img.save(dst, "TIFF", compression="tiff_lzw")
#         else:
#             img.save(dst, "PNG")


# def copy_or_convert_image(src: Path, out_images: Path, stem: str) -> str:
#     """
#     Write image under out_images; returns final file_name (may change .gif -> .jpg).
#     """
#     suf = src.suffix
#     if suf.lower() == ".gif":
#         name = f"{stem}.jpg"
#         dst = out_images / name
#         convert_gif_to_rgb_jpg(src, dst)
#         return name
#     name = src.name
#     dst = out_images / name
#     save_image_rgb(src, dst)
#     return name


# def process_gt_file(
#     gt_path: Path,
#     allowed: set[str],
#     mixed_policy: str,
# ) -> tuple[str, list[str], dict]:
#     """
#     Returns (status, out_lines, stats).

#     status:
#       - 'drop_image' : no valid allowed-script annotation in file
#       - 'keep'       : write out_lines (may be empty if something odd)
#     """
#     stats = {
#         "raw_lines": 0,
#         "kept": 0,
#         "removed": 0,
#         "relabeled_none": 0,
#         "skipped_invalid": 0,
#     }
#     raw = gt_path.read_text(encoding="utf-8-sig").splitlines()
#     parsed_rows: list[tuple[str, list[str] | None, str | None]] = []
#     for line in raw:
#         if not line.strip():
#             continue
#         stats["raw_lines"] += 1
#         parts, err = parse_gt_line(line)
#         parsed_rows.append((line, parts, err))

#     has_allowed = False
#     for _line, parts, err in parsed_rows:
#         if err or parts is None:
#             continue
#         script_key, valid = annotation_script_and_valid(parts)
#         if valid and script_key in allowed:
#             has_allowed = True
#             break

#     if not has_allowed:
#         return "drop_image", [], stats

#     out_lines: list[str] = []
#     for line, parts, err in parsed_rows:
#         if err or parts is None:
#             stats["skipped_invalid"] += 1
#             out_lines.append(line.strip())
#             continue
#         script_key, valid = annotation_script_and_valid(parts)
#         if not valid:
#             out_lines.append(line.strip())
#             stats["kept"] += 1
#             continue

#         if script_key in allowed:
#             out_lines.append(line_to_csv(parts))
#             stats["kept"] += 1
#             continue

#         if mixed_policy == "remove":
#             stats["removed"] += 1
#             continue

#         parts = list(parts)
#         parts[8] = "None"
#         out_lines.append(line_to_csv(parts))
#         stats["relabeled_none"] += 1
#         stats["kept"] += 1

#     return "keep", out_lines, stats


# def main():
#     args = parse_args()
#     image_dir = Path(args.images)
#     gt_dir = Path(args.gt)
#     allowed = {s.strip() for s in args.allowed_scripts.split(",") if s.strip()}
#     if not allowed:
#         raise SystemExit("--allowed_scripts must contain at least one script label")

#     if args.out_images:
#         out_images = Path(args.out_images)
#     else:
#         out_images = image_dir.parent / f"{image_dir.name}_script_filtered"
#     if args.out_gt:
#         out_gt = Path(args.out_gt)
#     else:
#         out_gt = gt_dir.parent / f"{gt_dir.name}_script_filtered"

#     if not image_dir.is_dir():
#         raise SystemExit(f"Not a directory: {image_dir}")
#     if not gt_dir.is_dir():
#         raise SystemExit(f"Not a directory: {gt_dir}")

#     summary = {
#         "gt_files": 0,
#         "images_dropped_no_allowed": 0,
#         "images_missing_file": 0,
#         "images_kept": 0,
#         "lines_removed": 0,
#         "lines_relabeled_none": 0,
#         "lines_kept": 0,
#         "invalid_gt_lines": 0,
#     }

#     if not args.dry_run:
#         out_images.mkdir(parents=True, exist_ok=True)
#         out_gt.mkdir(parents=True, exist_ok=True)

#     for gt_path in sorted(gt_dir.glob("*.txt")):
#         summary["gt_files"] += 1
#         stem = gt_path.stem
#         status, out_lines, st = process_gt_file(gt_path, allowed, args.mixed_policy)
#         summary["lines_removed"] += st["removed"]
#         summary["lines_relabeled_none"] += st["relabeled_none"]
#         summary["lines_kept"] += st["kept"]
#         summary["invalid_gt_lines"] += st["skipped_invalid"]

#         if status == "drop_image":
#             summary["images_dropped_no_allowed"] += 1
#             if args.dry_run:
#                 print(f"DROP (no allowed script): {stem}")
#             continue

#         found = find_image_with_prefix(image_dir, stem)
#         if found is None:
#             summary["images_missing_file"] += 1
#             print(f"WARN: GT {gt_path.name} has allowed script but no image for stem '{stem}'")
#             continue

#         src_img, _ext = found
#         if args.dry_run:
#             print(f"KEEP: {stem} -> {out_images / src_img.name} (policy={args.mixed_policy})")
#             summary["images_kept"] += 1
#             continue

#         final_name = copy_or_convert_image(src_img, out_images, stem)
#         out_gt_path = out_gt / f"{stem}.txt"
#         text = "\n".join(out_lines)
#         if text and not text.endswith("\n"):
#             text += "\n"
#         out_gt_path.write_text(text, encoding="utf-8", newline="\n")
#         if final_name != src_img.name:
#             pass
#         summary["images_kept"] += 1

#     print("\n=== filter_icdar2019_by_script ===")
#     print(f"allowed_scripts     : {sorted(allowed)}")
#     print(f"mixed_policy        : {args.mixed_policy}")
#     print(f"in  images / gt     : {image_dir} / {gt_dir}")
#     if not args.dry_run:
#         print(f"out images / gt     : {out_images} / {out_gt}")
#     print(f"GT files scanned      : {summary['gt_files']}")
#     print(f"Images dropped        : {summary['images_dropped_no_allowed']} (no allowed-script ann)")
#     print(f"Images missing file   : {summary['images_missing_file']}")
#     print(f"Images written        : {summary['images_kept']}")
#     print(f"GT lines kept         : {summary['lines_kept']}")
#     print(f"GT lines removed      : {summary['lines_removed']}")
#     print(f"GT lines -> None      : {summary['lines_relabeled_none']}")
#     print(f"Invalid/unparsed lines: {summary['invalid_gt_lines']}")
#     if args.dry_run:
#         print("\n(dry run: no files written)")


# if __name__ == "__main__":
#     main()



import json
from pathlib import Path
import shutil
from collections import defaultdict
import string

class ICDARLatinFilter:
    def __init__(self, image_dir, gt_path, output_dir, output_json_path):
        self.image_dir = Path(image_dir)
        self.gt_path = Path(gt_path)
        self.output_dir = Path(output_dir)
        self.output_json_path = Path(output_json_path)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load JSON
        print(f"Loading GT from: {self.gt_path}")
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        
        print(f"Loaded {len(self.gt_data)} images")
        
    def is_latin_text(self, text):
        """Check if text is Latin-based (ASCII + common Latin diacritics)"""
        if not text or text.strip() == "":
            return False
        
        # Latin Unicode ranges
        latin_ranges = [
            (0x0000, 0x007F),  # Basic Latin
            (0x0080, 0x00FF),  # Latin-1 Supplement
            (0x0100, 0x017F),  # Latin Extended-A
            (0x0180, 0x024F),  # Latin Extended-B
            (0x1E00, 0x1EFF),  # Latin Extended Additional
        ]
        
        # Non-Latin scripts to exclude
        non_latin_blocks = [
            (0x0400, 0x04FF),  # Cyrillic
            (0x0530, 0x058F),  # Armenian
            (0x0590, 0x05FF),  # Hebrew
            (0x0600, 0x06FF),  # Arabic
            (0x0E00, 0x0E7F),  # Thai
            (0x1100, 0x11FF),  # Hangul
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Extension A
            (0x20000, 0x2A6DF), # CJK Extension B
            (0x2B750, 0x2B81F), # CJK Extension E
        ]
        
        # Check each character
        for char in text:
            if char.isprintable() and not char.isspace():
                code = ord(char)
                
                # If character is in non-Latin block, text is not Latin
                if any(start <= code <= end for start, end in non_latin_blocks):
                    return False
        
        # If no non-Latin characters found, it's Latin
        return True
    
    def filter_annotations(self):
        """Filter out non-Latin annotations and images"""
        print("\n" + "="*60)
        print("FILTERING DATASET")
        print("="*60)
        
        filtered_data = {}
        stats = {
            'total_images_original': len(self.gt_data),
            'total_annotations_original': 0,
            'images_kept': 0,
            'images_removed': 0,
            'annotations_kept': 0,
            'annotations_removed': 0,
            'latin_annotations': 0,
            'non_latin_annotations': 0,
            'empty_annotations': 0
        }
        
        # Process each image
        for img_id, annotations in self.gt_data.items():
            stats['total_annotations_original'] += len(annotations)
            
            # Filter annotations
            kept_annotations = []
            for ann in annotations:
                transcription = ann.get('transcription', '')
                
                # Check if annotation is Latin
                if self.is_latin_text(transcription):
                    kept_annotations.append(ann)
                    stats['latin_annotations'] += 1
                else:
                    stats['non_latin_annotations'] += 1
                    if not transcription or transcription.strip() == "":
                        stats['empty_annotations'] += 1
            
            # Keep image only if it has at least one Latin annotation
            if kept_annotations:
                filtered_data[img_id] = kept_annotations
                stats['images_kept'] += 1
                stats['annotations_kept'] += len(kept_annotations)
            else:
                stats['images_removed'] += 1
        
        stats['annotations_removed'] = stats['total_annotations_original'] - stats['annotations_kept']
        
        self.filtered_data = filtered_data
        self.stats = stats
        
        return filtered_data, stats
    
    def copy_images(self):
        """Copy only the kept images to output directory"""
        print("\n" + "="*60)
        print("COPYING FILTERED IMAGES")
        print("="*60)
        
        copied_count = 0
        missing_count = 0
        missing_images = []
        
        for img_id in self.filtered_data.keys():
            # Try to find image with common extensions
            img_found = False
            for ext in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']:
                src_path = self.image_dir / f"{img_id}{ext}"
                if src_path.exists():
                    dst_path = self.output_dir / f"{img_id}{ext}"
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    img_found = True
                    break
            
            if not img_found:
                missing_count += 1
                missing_images.append(img_id)
        
        print(f"✓ Copied {copied_count} images")
        if missing_count > 0:
            print(f"⚠️  Warning: {missing_count} images not found in source directory")
            print(f"   Missing IDs (first 10): {missing_images[:10]}")
        
        return copied_count, missing_images
    
    def save_filtered_json(self):
        """Save the filtered GT to a new JSON file"""
        print("\n" + "="*60)
        print("SAVING FILTERED GROUND TRUTH")
        print("="*60)
        
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved filtered GT to: {self.output_json_path}")
        print(f"  File size: {self.output_json_path.stat().st_size / 1024:.2f} KB")
    
    def print_statistics(self):
        """Print filtering statistics"""
        print("\n" + "="*60)
        print("FILTERING STATISTICS")
        print("="*60)
        
        print(f"\n📊 Original Dataset:")
        print(f"  - Images: {self.stats['total_images_original']:,}")
        print(f"  - Annotations: {self.stats['total_annotations_original']:,}")
        
        print(f"\n📊 Filtered Dataset (Latin only):")
        print(f"  - Images kept: {self.stats['images_kept']:,} ({self.stats['images_kept']/self.stats['total_images_original']*100:.1f}%)")
        print(f"  - Images removed: {self.stats['images_removed']:,} ({self.stats['images_removed']/self.stats['total_images_original']*100:.1f}%)")
        print(f"  - Annotations kept: {self.stats['annotations_kept']:,} ({self.stats['annotations_kept']/self.stats['total_annotations_original']*100:.1f}%)")
        print(f"  - Annotations removed: {self.stats['annotations_removed']:,} ({self.stats['annotations_removed']/self.stats['total_annotations_original']*100:.1f}%)")
        
        print(f"\n📝 Annotation Breakdown:")
        print(f"  - Latin annotations kept: {self.stats['latin_annotations']:,}")
        print(f"  - Non-Latin annotations removed: {self.stats['non_latin_annotations']:,}")
        print(f"  - Empty annotations removed: {self.stats['empty_annotations']:,}")
        
        # Sample of kept images
        print(f"\n📸 Sample of kept images (first 10):")
        sample_ids = list(self.filtered_data.keys())[:10]
        for img_id in sample_ids:
            num_annots = len(self.filtered_data[img_id])
            print(f"  - {img_id}: {num_annots} Latin annotations")
    
    def create_verification_report(self):
        """Create a verification report comparing original vs filtered"""
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
        print("="*60)
        
        # Check if filtering worked as expected
        verification_passed = True
        
        # Verify no non-Latin annotations remain
        print("\n🔍 Verifying filtered dataset...")
        non_latin_found = 0
        
        for img_id, annotations in self.filtered_data.items():
            for ann in annotations:
                transcription = ann.get('transcription', '')
                if not self.is_latin_text(transcription) and transcription.strip() != "":
                    non_latin_found += 1
                    print(f"  ⚠️  Found non-Latin in {img_id}: '{transcription[:50]}'")
        
        if non_latin_found == 0:
            print("  ✓ No non-Latin annotations found in filtered dataset")
        else:
            print(f"  ❌ Found {non_latin_found} non-Latin annotations in filtered dataset!")
            verification_passed = False
        
        # Verify images exist for all kept IDs
        print("\n🔍 Verifying image files...")
        missing_in_output = []
        
        for img_id in self.filtered_data.keys():
            found = False
            for ext in ['.jpg', '.png', '.jpeg', '.tiff', '.bmp']:
                if (self.output_dir / f"{img_id}{ext}").exists():
                    found = True
                    break
            if not found:
                missing_in_output.append(img_id)
        
        if missing_in_output:
            print(f"  ⚠️  Warning: {len(missing_in_output)} images missing in output directory")
            verification_passed = False
        else:
            print("  ✓ All images copied successfully")
        
        return verification_passed
    
    def run_filtering(self):
        """Run the complete filtering pipeline"""
        print("\n" + "="*80)
        print("STARTING LATIN-ONLY DATASET FILTERING")
        print("="*80)
        
        # Filter annotations
        filtered_data, stats = self.filter_annotations()
        
        # Print statistics
        self.print_statistics()
        
        # Save filtered JSON
        self.save_filtered_json()
        
        # Copy images
        copied_count, missing_images = self.copy_images()
        
        # Create verification report
        verification_passed = self.create_verification_report()
        
        # Final summary
        print("\n" + "="*80)
        print("FILTERING COMPLETE")
        print("="*80)
        
        if verification_passed and len(filtered_data) > 0:
            print("\n✅ Success! Latin-only dataset created:")
            print(f"   - Output directory: {self.output_dir}")
            print(f"   - Output JSON: {self.output_json_path}")
            print(f"   - Images: {len(filtered_data):,}")
            print(f"   - Annotations: {stats['annotations_kept']:,}")
        else:
            print("\n⚠️  Filtering completed with warnings. Please check the verification report.")
        
        return filtered_data, stats


def filter_both_splits(train_config, test_config):
    """Helper function to filter both train and test splits"""
    print("\n" + "="*80)
    print("FILTERING BOTH TRAIN AND TEST SPLITS")
    print("="*80)
    
    # Filter training set
    print("\n🔵 PROCESSING TRAINING SET")
    train_filter = ICDARLatinFilter(
        image_dir=train_config['image_dir'],
        gt_path=train_config['gt_path'],
        output_dir=train_config['output_dir'],
        output_json_path=train_config['output_json']
    )
    train_filtered, train_stats = train_filter.run_filtering()
    
    # Filter test set
    print("\n🟢 PROCESSING TEST SET")
    test_filter = ICDARLatinFilter(
        image_dir=test_config['image_dir'],
        gt_path=test_config['gt_path'],
        output_dir=test_config['output_dir'],
        output_json_path=test_config['output_json']
    )
    test_filtered, test_stats = test_filter.run_filtering()
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL FILTERING SUMMARY")
    print("="*80)
    
    print(f"\n📊 Training Set Reduction:")
    print(f"  - Images: {train_stats['total_images_original']:,} → {train_stats['images_kept']:,} ({train_stats['images_kept']/train_stats['total_images_original']*100:.1f}%)")
    print(f"  - Annotations: {train_stats['total_annotations_original']:,} → {train_stats['annotations_kept']:,} ({train_stats['annotations_kept']/train_stats['total_annotations_original']*100:.1f}%)")
    
    print(f"\n📊 Test Set Reduction:")
    print(f"  - Images: {test_stats['total_images_original']:,} → {test_stats['images_kept']:,} ({test_stats['images_kept']/test_stats['total_images_original']*100:.1f}%)")
    print(f"  - Annotations: {test_stats['total_annotations_original']:,} → {test_stats['annotations_kept']:,} ({test_stats['annotations_kept']/test_stats['total_annotations_original']*100:.1f}%)")
    
    return train_filtered, test_filtered


# Example usage
if __name__ == "__main__":
    # For filtering a single JSON (before split)
    # Use this if you have one combined dataset
    SINGLE_DATASET_MODE = True
    
    if SINGLE_DATASET_MODE:
        # Single dataset mode (before splitting)
        IMAGE_DIR = r"E:\PFE\ICDAR2019\train_full_images_1"
        GT_PATH = r"E:\PFE\ICDAR2019\train_full_labels.json"
        OUTPUT_DIR = r"E:\PFE\ICDAR2019\filtered_latin"
        OUTPUT_JSON = r"E:\PFE\ICDAR2019\filtered_latin\filtered_annotations.json"
        
        filter_tool = ICDARLatinFilter(IMAGE_DIR, GT_PATH, OUTPUT_DIR, OUTPUT_JSON)
        filtered_data, stats = filter_tool.run_filtering()
    
    else:
        # For filtering already split datasets (train/test)
        train_config = {
            'image_dir': r"E:\PFE\ICDAR2019\split\train_img",
            'gt_path': r"E:\PFE\ICDAR2019\split\train.json",
            'output_dir': r"E:\PFE\ICDAR2019\filtered_latin\train_img",
            'output_json': r"E:\PFE\ICDAR2019\filtered_latin\train.json"
        }
        
        test_config = {
            'image_dir': r"E:\PFE\ICDAR2019\split\test_img",
            'gt_path': r"E:\PFE\ICDAR2019\split\test.json",
            'output_dir': r"E:\PFE\ICDAR2019\filtered_latin\test_img",
            'output_json': r"E:\PFE\ICDAR2019\filtered_latin\test.json"
        }
        
        # Filter both splits
        train_filtered, test_filtered = filter_both_splits(train_config, test_config)