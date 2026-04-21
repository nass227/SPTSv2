import json
import pandas as pd
import os
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================
train_json_path = r"E:\PFE\ICDAR2019\train.json"
test_json_path = r"E:\PFE\ICDAR2019\split_dataset_cleaned\test.json"
output_dir = r"E:\PFE\ICDAR_cleaned\eda_results"
os.makedirs(output_dir, exist_ok=True)

# SEARCH OPTIONS - Uncomment/Modify as needed
SEARCH_MODE = "image_name"  # Options: "annotation_text", "image_name", "both"
TARGET_TEXT = "Rääääääääääääääääääääääää"  # For annotation text search
TARGET_IMAGE_NAME = "3503"  # For image name search (supports partial)
PARTIAL_MATCH = True  # If True, search for partial matches in image names

def load_json_safe(json_path):
    """Load JSON with encoding fallback"""
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except:
            continue
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)

def search_by_image_name(json_path, split_name, target_name, partial=True):
    """Find images by name (full or partial match)"""
    
    print(f"\n{'='*60}")
    print(f"Searching for image: '{target_name}' in {split_name.upper()} split")
    print(f"{'='*60}")
    
    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        return []
    
    data = load_json_safe(json_path)
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    # Create mapping for quick lookup
    ann_by_image = {}
    for ann in annotations:
        img_id = ann.get('image_id')
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)
    
    found_images = []
    
    for img in images:
        file_name = img.get('file_name', '')
        
        # Check if image matches search criteria
        if partial:
            if target_name.lower() in file_name.lower():
                found_images.append(img)
        else:
            if file_name == target_name:
                found_images.append(img)
    
    if not found_images:
        print(f"No images found matching: '{target_name}'")
        return []
    
    print(f"Found {len(found_images)} image(s) matching: '{target_name}'\n")
    
    for i, img in enumerate(found_images, 1):
        img_id = img.get('id')
        img_annotations = ann_by_image.get(img_id, [])
        
        print(f"\n--- Image {i} ---")
        print(f"Image ID: {img_id}")
        print(f"File name: {img.get('file_name')}")
        print(f"Dimensions: {img.get('width')} x {img.get('height')}")
        print(f"Number of annotations: {len(img_annotations)}")
        
        if img_annotations:
            print(f"\n  Annotations for this image:")
            for j, ann in enumerate(img_annotations[:5], 1):  # Show first 5
                rec_string = ann.get('rec_string', '')
                bbox = ann.get('bbox', [])
                print(f"    {j}. ID: {ann.get('id')}, Text: '{rec_string[:50]}', BBox: {bbox}")
            
            if len(img_annotations) > 5:
                print(f"    ... and {len(img_annotations) - 5} more annotations")
        else:
            print(f"\n  WARNING: No annotations found for this image!")
        
        print("-"*40)
    
    return found_images

def search_by_annotation_text(json_path, split_name, target_text):
    """Find all annotations with the target text"""
    
    print(f"\n{'='*60}")
    print(f"Searching for text: '{target_text}' in {split_name.upper()} split")
    print(f"{'='*60}")
    
    if not os.path.exists(json_path):
        print(f"ERROR: File not found: {json_path}")
        return []
    
    data = load_json_safe(json_path)
    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])
    
    found_annotations = []
    
    for ann in annotations:
        rec_string = ann.get('rec_string', '')
        if rec_string == target_text:
            found_annotations.append(ann)
    
    if not found_annotations:
        print(f"No annotations found with text: '{target_text}'")
        return []
    
    print(f"Found {len(found_annotations)} annotation(s) with text: '{target_text}'\n")
    
    for i, ann in enumerate(found_annotations, 1):
        print(f"\n--- Annotation {i} ---")
        print(f"Annotation ID: {ann.get('id')}")
        print(f"Image ID: {ann.get('image_id')}")
        print(f"Category ID: {ann.get('category_id')}")
        print(f"BBox: {ann.get('bbox')}")
        print(f"Area: {ann.get('area')}")
        print(f"iscrowd: {ann.get('iscrowd')}")
        print(f"Text: {ann.get('rec_string')}")
        
        # Get image info
        img_id = ann.get('image_id')
        if img_id in images:
            img = images[img_id]
            print(f"\nImage Info:")
            print(f"  File name: {img.get('file_name')}")
            print(f"  Image dimensions: {img.get('width')} x {img.get('height')}")
            print(f"  Image ID: {img.get('id')}")
        else:
            print(f"\nWarning: Image ID {img_id} not found in images list")
        
        # Show bezier points if available
        if 'bezier_pts' in ann:
            bezier = ann.get('bezier_pts')
            if bezier:
                print(f"\nBezier points (first 8): {bezier[:8] if len(bezier) > 8 else bezier}")
        
        print("-"*40)
    
    return found_annotations

def search_both(json_path, split_name, target_image, target_text, partial=True):
    """Search for both image name and annotation text"""
    
    print(f"\n{'='*60}")
    print(f"SEARCHING BOTH in {split_name.upper()} split")
    print(f"{'='*60}")
    
    results = {
        'images': [],
        'annotations': []
    }
    
    if target_image:
        results['images'] = search_by_image_name(json_path, split_name, target_image, partial)
    
    if target_text:
        results['annotations'] = search_by_annotation_text(json_path, split_name, target_text)
    
    return results

def save_results_to_file(results, output_path):
    """Save search results to a text file"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"SEARCH RESULTS\n")
        f.write(f"Mode: {SEARCH_MODE.upper()}\n")
        if SEARCH_MODE in ["annotation_text", "both"]:
            f.write(f"Target Text: '{TARGET_TEXT}'\n")
        if SEARCH_MODE in ["image_name", "both"]:
            f.write(f"Target Image: '{TARGET_IMAGE_NAME}'\n")
            f.write(f"Partial Match: {PARTIAL_MATCH}\n")
        f.write("="*60 + "\n\n")
        
        if 'train' in results:
            f.write("TRAIN SPLIT RESULTS\n")
            f.write("-"*40 + "\n")
            
            if SEARCH_MODE in ["image_name", "both"] and results['train'].get('images'):
                f.write(f"\nImages found: {len(results['train']['images'])}\n")
                for i, img in enumerate(results['train']['images'], 1):
                    f.write(f"\n  Image {i}:\n")
                    f.write(f"    ID: {img.get('id')}\n")
                    f.write(f"    File: {img.get('file_name')}\n")
                    f.write(f"    Size: {img.get('width')}x{img.get('height')}\n")
            
            if SEARCH_MODE in ["annotation_text", "both"] and results['train'].get('annotations'):
                f.write(f"\nAnnotations found: {len(results['train']['annotations'])}\n")
                for i, ann in enumerate(results['train']['annotations'], 1):
                    f.write(f"\n  Annotation {i}:\n")
                    f.write(f"    ID: {ann.get('id')}\n")
                    f.write(f"    Image ID: {ann.get('image_id')}\n")
                    f.write(f"    Text: {ann.get('rec_string')}\n")
                    f.write(f"    BBox: {ann.get('bbox')}\n")
        
        if 'test' in results:
            f.write("\n\nTEST SPLIT RESULTS\n")
            f.write("-"*40 + "\n")
            
            if SEARCH_MODE in ["image_name", "both"] and results['test'].get('images'):
                f.write(f"\nImages found: {len(results['test']['images'])}\n")
                for i, img in enumerate(results['test']['images'], 1):
                    f.write(f"\n  Image {i}:\n")
                    f.write(f"    ID: {img.get('id')}\n")
                    f.write(f"    File: {img.get('file_name')}\n")
                    f.write(f"    Size: {img.get('width')}x{img.get('height')}\n")
            
            if SEARCH_MODE in ["annotation_text", "both"] and results['test'].get('annotations'):
                f.write(f"\nAnnotations found: {len(results['test']['annotations'])}\n")
                for i, ann in enumerate(results['test']['annotations'], 1):
                    f.write(f"\n  Annotation {i}:\n")
                    f.write(f"    ID: {ann.get('id')}\n")
                    f.write(f"    Image ID: {ann.get('image_id')}\n")
                    f.write(f"    Text: {ann.get('rec_string')}\n")
                    f.write(f"    BBox: {ann.get('bbox')}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF SEARCH RESULTS\n")
        f.write("="*60 + "\n")

def main():
    print("="*60)
    print(f"SEARCH MODE: {SEARCH_MODE.upper()}")
    if SEARCH_MODE in ["annotation_text", "both"]:
        print(f"Searching for text: '{TARGET_TEXT}'")
    if SEARCH_MODE in ["image_name", "both"]:
        print(f"Searching for image: '{TARGET_IMAGE_NAME}' (Partial: {PARTIAL_MATCH})")
    print("="*60)
    
    results = {
        'train': {},
        'test': {}
    }
    
    # Search based on mode
    if SEARCH_MODE == "annotation_text":
        results['train']['annotations'] = search_by_annotation_text(train_json_path, 'train', TARGET_TEXT)
        results['test']['annotations'] = search_by_annotation_text(test_json_path, 'test', TARGET_TEXT)
    
    elif SEARCH_MODE == "image_name":
        results['train']['images'] = search_by_image_name(train_json_path, 'train', TARGET_IMAGE_NAME, PARTIAL_MATCH)
        results['test']['images'] = search_by_image_name(test_json_path, 'test', TARGET_IMAGE_NAME, PARTIAL_MATCH)
    
    elif SEARCH_MODE == "both":
        results['train'] = search_both(train_json_path, 'train', TARGET_IMAGE_NAME, TARGET_TEXT, PARTIAL_MATCH)
        results['test'] = search_both(test_json_path, 'test', TARGET_IMAGE_NAME, TARGET_TEXT, PARTIAL_MATCH)
    
    # Save results
    output_file = os.path.join(output_dir, f'search_results_{SEARCH_MODE}.txt')
    save_results_to_file(results, output_file)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Summary
    print(f"\nSUMMARY:")
    if SEARCH_MODE in ["annotation_text", "both"]:
        train_ann_count = len(results['train'].get('annotations', []))
        test_ann_count = len(results['test'].get('annotations', []))
        print(f"  Annotations:")
        print(f"    Train split: {train_ann_count}")
        print(f"    Test split: {test_ann_count}")
        print(f"    Total: {train_ann_count + test_ann_count}")
    
    if SEARCH_MODE in ["image_name", "both"]:
        train_img_count = len(results['train'].get('images', []))
        test_img_count = len(results['test'].get('images', []))
        print(f"  Images:")
        print(f"    Train split: {train_img_count}")
        print(f"    Test split: {test_img_count}")
        print(f"    Total: {train_img_count + test_img_count}")

if __name__ == "__main__":
    main()