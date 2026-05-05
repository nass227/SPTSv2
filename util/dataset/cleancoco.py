import json
import os
import shutil
from collections import defaultdict

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
INPUT_JSON_PATH = r"D:\ICDAR\annotations_test_all.json"
OUTPUT_JSON_PATH = r"D:\ICDAR\annotations_test_all_cleaned.json"
IMAGE_DIR = r"D:\ICDAR\test_images"  # Set to None if you don't want to process images
REMOVE_EMPTY_IMAGES = True  # Delete images that have no annotations after cleaning
CREATE_BACKUP = True  # Create backup of original JSON

# ============================================
# SCRIPT STARTS HERE
# ============================================

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_dataset(json_path, output_path, image_dir=None, remove_empty_images=True):
    """Remove annotations with '###' and optionally remove empty images"""
    
    print(f"Loading: {json_path}")
    data = load_json(json_path)
    
    # Statistics
    original_ann_count = len(data['annotations'])
    original_img_count = len(data['images'])
    
    # Track which images have valid annotations
    image_has_valid_ann = defaultdict(int)
    
    # Filter annotations
    valid_annotations = []
    removed_count = 0
    
    for ann in data['annotations']:
        rec_string = ann.get('rec_string', '')
        
        # Keep only annotations without '###'
        if '###' not in rec_string:
            valid_annotations.append(ann)
            image_has_valid_ann[ann['image_id']] += 1
        else:
            removed_count += 1
    
    # Filter images
    if remove_empty_images:
        valid_images = [img for img in data['images'] if image_has_valid_ann.get(img['id'], 0) > 0]
    else:
        valid_images = data['images']
    
    # Create cleaned dataset
    cleaned_data = {
        'images': valid_images,
        'annotations': valid_annotations,
        'categories': data.get('categories', [])
    }
    
    # Save cleaned JSON
    save_json(cleaned_data, output_path)
    print(f"Saved: {output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("CLEANING STATISTICS")
    print("="*50)
    print(f"Annotations: {original_ann_count:,} -> {len(valid_annotations):,}")
    print(f"  Removed: {removed_count:,} (contained '###')")
    print(f"Images: {original_img_count:,} -> {len(valid_images):,}")
    print(f"  Removed: {original_img_count - len(valid_images):,} (became empty)")
    print("="*50)
    
    # Delete image files if directory provided
    if image_dir and remove_empty_images and os.path.exists(image_dir):
        print(f"\nDeleting empty image files from: {image_dir}")
        
        # Get valid image IDs
        valid_image_ids = {img['id'] for img in valid_images}
        
        # Find and delete image files for removed images
        deleted_files = 0
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Try to extract image ID from filename
                name_without_ext = os.path.splitext(filename)[0]
                try:
                    img_id = int(''.join(filter(str.isdigit, name_without_ext)))
                    if img_id not in valid_image_ids:
                        file_path = os.path.join(image_dir, filename)
                        os.remove(file_path)
                        deleted_files += 1
                        print(f"  Deleted: {filename}")
                except ValueError:
                    # Skip files that don't have numeric IDs
                    pass
        
        print(f"\nDeleted {deleted_files} image files")
    
    return cleaned_data

def main():
    # Create backup if requested
    if CREATE_BACKUP and os.path.exists(INPUT_JSON_PATH):
        backup_path = INPUT_JSON_PATH.replace('.json', '_backup.json')
        if not os.path.exists(backup_path):
            shutil.copy2(INPUT_JSON_PATH, backup_path)
            print(f"Backup created: {backup_path}\n")
    
    # Clean the dataset
    clean_dataset(
        json_path=INPUT_JSON_PATH,
        output_path=OUTPUT_JSON_PATH,
        image_dir=IMAGE_DIR if IMAGE_DIR else None,
        remove_empty_images=REMOVE_EMPTY_IMAGES
    )
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()