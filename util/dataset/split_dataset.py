"""
Split dataset into train/test sets (80/20) with randomized split.

Input:
  --annotations : JSON file with annotations (COCO format or your original format).
                  For COCO, only image ids present in annotations[] are split; entries
                  in images[] with zero annotations (e.g. after cleaning) are ignored.
  --images-dir  : Directory containing all images
  --output-dir  : Output directory for split datasets
  --train-ratio : Train ratio (default: 0.8)
  --seed        : Random seed for reproducibility (default: 42)

Output:
  - train_images/     : Symbolic links or copies of training images
  - test_images/      : Symbolic links or copies of test images
  - train_annotations.json
  - test_annotations.json
  - split_info.json   : Contains split details and statistics

Usage:
  python split_dataset.py  --annotations train_full_labels_cleaned.json  --images-dir E:/PFE/ICDAR2019/images --output-dir E:/PFE/ICDAR2019/split_dataset  --train-ratio 0.8  --seed 42  --copy-images    # Use --copy-images to copy instead of symlink
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP", ".TIFF"}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def find_image_file(image_dir: Path, file_name: str) -> Path | None:
    """Find image file by its filename (from COCO file_name field)"""
    # Try exact filename match
    img_path = image_dir / file_name
    if img_path.exists():
        return img_path
    
    # If not found, try without extension (in case extension mismatch)
    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        img_path = image_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    
    return None

def get_image_info_from_coco(data: dict) -> List[Tuple[int, str]]:
    """Extract (image_id, file_name) pairs from COCO format"""
    if "images" not in data:
        return []
    
    return [(img["id"], img["file_name"]) for img in data["images"]]


def coco_annotated_image_ids(data: dict) -> Set[str]:
    """Image ids that have at least one annotation (after cleaning, many COCO files still list all images)."""
    return {str(ann["image_id"]) for ann in data.get("annotations", [])}

def get_image_id_from_annotation(annotation_format: str, data: dict) -> List[str]:
    """Extract image IDs based on annotation format"""
    
    # Case 1: COCO format {"images": [...], "annotations": [...]}
    if "images" in data:
        # Return the actual image IDs (not filenames)
        return [str(img["id"]) for img in data["images"]]
    
    # Case 2: Original format {"gt_1234": [...], "gt_5678": [...]}
    elif all(k.startswith("gt_") for k in list(data.keys())[:10]):
        return list(data.keys())
    
    # Case 3: Any dict where keys are image IDs
    else:
        return list(data.keys())

def filter_annotations_by_images(data: dict, keep_image_ids: Set[str], 
                                  annotation_format: str) -> dict:
    """Filter annotations to only keep specified images by their ID"""
    
    # COCO format
    if "images" in data:
        # Filter images by their id field
        filtered_images = [img for img in data["images"] 
                          if str(img["id"]) in keep_image_ids]
        
        # Filter annotations by image_id
        filtered_anns = [ann for ann in data["annotations"] 
                        if str(ann["image_id"]) in keep_image_ids]
        
        return {
            "images": filtered_images,
            "annotations": filtered_anns,
            "categories": data.get("categories", [{"id": 1, "name": "text"}])
        }
    
    # Original format (gt_1234: [...])
    else:
        return {img_id: data[img_id] for img_id in keep_image_ids if img_id in data}

def copy_or_link_image(
    src_path: Path, dst_path: Path, copy_mode: bool = False
) -> Tuple[bool, Optional[str]]:
    """Copy or create symbolic link for image file. Returns (ok, error_message)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if copy_mode:
            shutil.copy2(src_path, dst_path)
        else:
            # Try symlink, fallback to copy if symlink fails (Windows without admin)
            try:
                dst_path.symlink_to(src_path)
            except (OSError, NotImplementedError):
                shutil.copy2(src_path, dst_path)
        return True, None
    except OSError as exc:
        # PermissionError reading source, locked files, AV interference, etc.
        try:
            dst_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, str(exc)

# ---------------------------------------------------------------------------
# Main splitting function
# ---------------------------------------------------------------------------
def split_dataset(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy_images: bool = False,
) -> Dict:
    """Split dataset into train and test sets"""
    
    print(f"\n{'='*60}")
    print(f"SPLITTING DATASET")
    print(f"{'='*60}")
    print(f"Annotations: {annotations_path}")
    print(f"Images dir:  {images_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Train ratio: {train_ratio}")
    print(f"Test ratio:  {1-train_ratio}")
    print(f"Random seed: {seed}")
    print(f"Copy mode:   {copy_images}")
    print(f"{'='*60}\n")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    output_dir = Path(output_dir)
    train_img_dir = output_dir / "train_images"
    test_img_dir = output_dir / "test_images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Detect annotation format
    annotation_format = "coco" if "images" in data else "original"
    print(f"Annotation format: {annotation_format}")
    
    # For COCO format, build mapping from image_id to file_name
    if annotation_format == "coco":
        annotated_ids = coco_annotated_image_ids(data)
        image_info_all = get_image_info_from_coco(data)
        # Only split/copy images that still have annotations (images[] may list the full corpus)
        image_info = [
            (img_id, fn)
            for img_id, fn in image_info_all
            if str(img_id) in annotated_ids
        ]
        skipped_no_annotations = len(image_info_all) - len(image_info)
        ids_in_anns_not_in_images = annotated_ids - {str(i) for i, _ in image_info}

        print(f"Total image entries in JSON (images[]): {len(image_info_all)}")
        print(f"Images with at least one annotation: {len(image_info)}")
        if skipped_no_annotations:
            print(
                f"  (Skipping {skipped_no_annotations} image(s) listed in images[] "
                f"but with no annotations — e.g. removed during cleaning.)"
            )
        if ids_in_anns_not_in_images:
            print(
                f"  [WARN] {len(ids_in_anns_not_in_images)} image_id(s) appear in annotations "
                f"but not in images[] (cannot resolve file_name); those samples are omitted."
            )

        if image_info:
            print(f"Sample image info: id={image_info[0][0]}, file_name={image_info[0][1]}")
        
        # Get all image IDs for splitting
        all_image_ids = [str(img_id) for img_id, _ in image_info]
        
        # Create mapping for quick lookup (all listed images, for any edge case)
        id_to_filename = {str(img_id): file_name for img_id, file_name in image_info_all}
    else:
        all_image_ids = list(data.keys())
        print(f"Total images in JSON: {len(all_image_ids)}")
        id_to_filename = {img_id: f"{img_id}.jpg" for img_id in all_image_ids}  # Guess filename
    
    if not all_image_ids:
        print("\nError: No images to split (no samples with annotations).")
        return {"error": "no_images_to_split", "annotation_format": annotation_format}
    
    # Randomly split image IDs
    shuffled_ids = all_image_ids.copy()
    random.shuffle(shuffled_ids)
    
    split_idx = int(len(shuffled_ids) * train_ratio)
    train_ids = set(shuffled_ids[:split_idx])
    test_ids = set(shuffled_ids[split_idx:])
    
    n_total = len(all_image_ids)
    print(f"\nSplit statistics:")
    print(f"  Train images: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"  Test images:  {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    
    # Verify images exist and copy/link them
    print("\nProcessing images...")
    print(f"Looking for images in: {images_dir}")
    
    # List a few sample images to debug
    sample_images = list(images_dir.glob("*.jpg"))[:5]
    if sample_images:
        print(f"Sample images found: {[f.name for f in sample_images]}")
    
    train_images_found = []
    test_images_found = []
    missing_images = []
    
    # Process training images
    print(f"\nProcessing {len(train_ids)} training images...")
    for i, img_id in enumerate(train_ids):
        # Get the actual filename from COCO JSON
        file_name = id_to_filename.get(img_id)
        if not file_name:
            missing_images.append(('train', img_id, 'no_filename'))
            continue
        
        # Find the image file
        img_path = find_image_file(images_dir, file_name)
        if img_path:
            dst_path = train_img_dir / img_path.name
            ok, err = copy_or_link_image(img_path, dst_path, copy_images)
            if ok:
                train_images_found.append(img_id)
            else:
                missing_images.append(("train", img_id, f"{file_name} [copy: {err}]"))
                if i < 10:
                    print(f"  [WARN] Could not copy/link ID {img_id} ({file_name}): {err}")
        else:
            missing_images.append(('train', img_id, file_name))
            if i < 10:  # Print first 10 missing
                print(f"  [WARN] Missing image for ID {img_id}: {file_name}")
        
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(train_ids)} training images...")
    
    # Process test images
    print(f"\nProcessing {len(test_ids)} test images...")
    for i, img_id in enumerate(test_ids):
        # Get the actual filename from COCO JSON
        file_name = id_to_filename.get(img_id)
        if not file_name:
            missing_images.append(('test', img_id, 'no_filename'))
            continue
        
        # Find the image file
        img_path = find_image_file(images_dir, file_name)
        if img_path:
            dst_path = test_img_dir / img_path.name
            ok, err = copy_or_link_image(img_path, dst_path, copy_images)
            if ok:
                test_images_found.append(img_id)
            else:
                missing_images.append(("test", img_id, f"{file_name} [copy: {err}]"))
                if i < 10:
                    print(f"  [WARN] Could not copy/link ID {img_id} ({file_name}): {err}")
        else:
            missing_images.append(('test', img_id, file_name))
            if i < 10:  # Print first 10 missing
                print(f"  [WARN] Missing image for ID {img_id}: {file_name}")
        
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(test_ids)} test images...")
    
    # Filter annotations to keep only found images
    print("\nFiltering annotations...")
    train_data = filter_annotations_by_images(data, set(train_images_found), annotation_format)
    test_data = filter_annotations_by_images(data, set(test_images_found), annotation_format)
    
    # Calculate annotation counts
    if annotation_format == "coco":
        train_ann_count = len(train_data.get("annotations", []))
        test_ann_count = len(test_data.get("annotations", []))
    else:
        train_ann_count = sum(len(v) for v in train_data.values())
        test_ann_count = sum(len(v) for v in test_data.values())
    
    # Save split annotations
    print("Saving split annotations...")
    train_json_path = output_dir / "train_annotations.json"
    test_json_path = output_dir / "test_annotations.json"
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Save split info
    split_info = {
        "split_config": {
            "train_ratio": train_ratio,
            "test_ratio": 1 - train_ratio,
            "random_seed": seed,
            "copy_images": copy_images
        },
        "statistics": {
            "total_images_json": len(all_image_ids),
            "train_images_found": len(train_images_found),
            "test_images_found": len(test_images_found),
            "missing_images": len([m for m in missing_images if m[0] in ['train', 'test']]),
            "train_annotations": train_ann_count,
            "test_annotations": test_ann_count
        },
        "files": {
            "train_images_dir": str(train_img_dir),
            "test_images_dir": str(test_img_dir),
            "train_annotations": str(train_json_path),
            "test_annotations": str(test_json_path)
        }
    }
    
    if missing_images:
        split_info["missing_images"] = missing_images[:100]  # Save first 100
    
    info_path = output_dir / "split_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SPLIT COMPLETE!")
    print(f"{'='*60}")
    print(f"Images in JSON:    {len(all_image_ids)}")
    print(f"Train images:      {len(train_images_found)}/{len(train_ids)}")
    print(f"Test images:       {len(test_images_found)}/{len(test_ids)}")
    print(f"Train annotations: {train_ann_count}")
    print(f"Test annotations:  {test_ann_count}")
    print(f"Missing images:    {len([m for m in missing_images if m[0] in ['train', 'test']])}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - train_images/")
    print(f"  - test_images/")
    print(f"  - train_annotations.json")
    print(f"  - test_annotations.json")
    print(f"  - split_info.json")
    print(f"{'='*60}\n")
    
    return split_info

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test sets (80/20 randomized)"
    )
    parser.add_argument("--annotations", "-a", required=True,
                        help="Path to annotations JSON file")
    parser.add_argument("--images-dir", "-i", required=True,
                        help="Directory containing all images")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of creating symlinks")
    return parser.parse_args()

def main():
    args = parse_args()
    
    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return 1
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    if not 0 < args.train_ratio < 1:
        print(f"Error: train_ratio must be between 0 and 1, got {args.train_ratio}")
        return 1
    
    # Run split
    split_dataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        copy_images=args.copy_images
    )
    
    return 0

if __name__ == "__main__":
    exit(main())