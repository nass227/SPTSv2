import os
import json
import csv
from PIL import Image
import unicodedata
from pathlib import Path

# paths for ICDAR 2019 dataset 
IMAGE_DIR = r"../Data/ICDAR2019/test_imgs" 
GT_DIR = r"../Data/ICDAR2019/test_gt"
OUTPUT_JSON = "../Data/ICDAR2019/test.json"

MAX_TEXT_LEN = 25
NBINS = 1000

# Character set from SPTS v2 (97 characters)
base_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
characters = base_chars + punctuation

PAD_TOKEN = len(characters)
SOS_TOKEN = len(characters) + 1
EOS_TOKEN = len(characters) + 2

def normalize_text(text):
    normalized = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized if ord(c) < 128)

def encode_text(text):
    text = normalize_text(text)
    encoded = [SOS_TOKEN]
    
    for c in text:
        if c in characters:
            encoded.append(characters.index(c))
        else:
            encoded.append(characters.index(' ') if ' ' in characters else 0)
    
    if len(encoded) > MAX_TEXT_LEN:
        encoded = encoded[:MAX_TEXT_LEN]
    
    while len(encoded) < MAX_TEXT_LEN:
        encoded.append(PAD_TOKEN)
    
    encoded.append(EOS_TOKEN)
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

# Diagnostic counters
diagnostics = {
    'gt_files_found': 0,
    'images_found': 0,
    'gif_converted': 0,
    'images_missing': [],
    'images_corrupted': [],
    'gt_files_empty': [],
    'total_annotations': 0,
    'valid_annotations': 0,
    'invalid_bbox': 0,
    'no_text': 0,
    'wrong_script': 0,
    'parsing_errors': 0,
}

# First, get all GT files and match with images
gt_files = [f for f in os.listdir(GT_DIR) if f.endswith(".txt")]
diagnostics['gt_files_found'] = len(gt_files)

print(f"Found {len(gt_files)} GT files")
print("\nScanning for images and converting GIFs to JPG...")

# Include all common image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF', '.bmp', '.tif', '.tiff']
valid_images = []

for gt_file in gt_files:
    img_name_base = gt_file.replace(".txt", "")
    
    img_found = False
    original_img_path = None
    original_ext = None
    
    # First, check if any image exists
    for ext in image_extensions:
        img_path = Path(IMAGE_DIR) / (img_name_base + ext)
        if img_path.exists():
            img_found = True
            original_img_path = img_path
            original_ext = ext
            break
    
    if not img_found:
        diagnostics['images_missing'].append(gt_file)
        continue
    
    # If it's a GIF, convert it to JPG
    if original_ext.lower() == '.gif':
        jpg_path = convert_gif_to_jpg(original_img_path)
        if jpg_path and jpg_path.exists():
            # Use the converted JPG
            valid_images.append((gt_file, jpg_path.name, jpg_path))
            diagnostics['gif_converted'] += 1
        else:
            # If conversion fails, try to use original GIF
            valid_images.append((gt_file, original_img_path.name, original_img_path))
            print(f"  Warning: Using original GIF {original_img_path.name}")
    else:
        # Not a GIF, use as is
        valid_images.append((gt_file, original_img_path.name, original_img_path))

print(f"\nFound {len(valid_images)} valid image-GT pairs")
print(f"Converted {diagnostics['gif_converted']} GIFs to JPG")
print(f"Missing images for {len(diagnostics['images_missing'])} GT files")

if diagnostics['images_missing']:
    print("\nFirst 10 missing images:")
    for missing in diagnostics['images_missing'][:10]:
        print(f"  {missing}")

images = []
annotations = []

image_id = 1
ann_id = 1

debug = {
    'total_lines': 0,
    'kept': 0,
    'scripts_seen': set(),
    'scripts_filtered': set(),
}

# Process only valid pairs
for gt_file, img_name, img_path in valid_images:
    try:
        # Open and validate image
        with Image.open(img_path) as img:
            # Handle different image modes properly
            if img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode in ('RGBA', 'LA'):
                # Convert to RGB by compositing on white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            w, h = img.size
        
        diagnostics['images_found'] += 1
        
        images.append({
            "id": image_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        # Process GT file
        gt_full_path = os.path.join(GT_DIR, gt_file)
        
        # Check if GT file is empty
        if os.path.getsize(gt_full_path) == 0:
            diagnostics['gt_files_empty'].append(gt_file)
            image_id += 1
            continue
        
        with open(gt_full_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            
            if len(lines) == 0:
                diagnostics['gt_files_empty'].append(gt_file)
                image_id += 1
                continue
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                debug['total_lines'] += 1
                diagnostics['total_annotations'] += 1
                
                try:
                    parts = next(csv.reader([line]))
                    
                    if len(parts) < 10:
                        diagnostics['parsing_errors'] += 1
                        continue
                    
                    # Parse coordinates
                    x1 = float(parts[0])
                    y1 = float(parts[1])
                    x2 = float(parts[2])
                    y2 = float(parts[3])
                    x3 = float(parts[4])
                    y3 = float(parts[5])
                    x4 = float(parts[6])
                    y4 = float(parts[7])
                    
                    script = parts[8].strip() if len(parts) > 8 else "Unknown"
                    text = parts[9].strip() if len(parts) > 9 else ""
                    
                    debug['scripts_seen'].add(script)
                    
                    # Skip if no text
                    if not text or text == "###":
                        diagnostics['no_text'] += 1
                        continue
                    
                    # Convert to SPTS v2 format
                    bbox, bezier_pts = quad_to_bbox_and_bezier(x1, y1, x2, y2, x3, y3, x4, y4)
                    
                    # Validate bbox
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        diagnostics['invalid_bbox'] += 1
                        continue
                    
                    # Encode text
                    encoded_text = encode_text(text)
                    
                    # Create annotation in SPTS v2 format
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,  # [x, y, width, height]
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "rec": encoded_text,
                        "bezier_pts": bezier_pts  # 16 values
                    })
                    
                    debug['kept'] += 1
                    diagnostics['valid_annotations'] += 1
                    ann_id += 1
                    
                except Exception as e:
                    diagnostics['parsing_errors'] += 1
                    continue
        
        image_id += 1
        
    except Exception as e:
        diagnostics['images_corrupted'].append((img_name, str(e)))
        print(f"Error processing image {img_name}: {e}")
        continue

# SPTS v2 expects exactly this structure
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "text"}]
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco_format, f)

# Print detailed diagnostics
print("\n" + "="*60)
print("DIAGNOSTICS REPORT")
print("="*60)
print(f"\nFILE COUNTS:")
print(f"  GT files found: {diagnostics['gt_files_found']}")
print(f"  Valid image-GT pairs: {len(valid_images)}")
print(f"  Images successfully processed: {diagnostics['images_found']}")
print(f"  GIFs converted to JPG: {diagnostics['gif_converted']}")
print(f"  Images missing: {len(diagnostics['images_missing'])}")
print(f"  Images corrupted: {len(diagnostics['images_corrupted'])}")
print(f"  Empty GT files: {len(diagnostics['gt_files_empty'])}")

print(f"\nANNOTATION STATISTICS:")
print(f"  Total annotations in GT files: {diagnostics['total_annotations']}")
print(f"  Valid annotations in JSON: {diagnostics['valid_annotations']}")
print(f"  Invalid bbox: {diagnostics['invalid_bbox']}")
print(f"  No text (###): {diagnostics['no_text']}")
print(f"  Parsing errors: {diagnostics['parsing_errors']}")

print(f"\nOUTPUT SUMMARY:")
print(f"  Images in JSON: {len(images)}")
print(f"  Annotations in JSON: {len(annotations)}")
print(f"  Scripts seen: {debug['scripts_seen']}")

if diagnostics['images_missing']:
    print(f"\n⚠ Missing images (first 10):")
    for missing in diagnostics['images_missing'][:10]:
        print(f"  {missing}")

if diagnostics['images_corrupted']:
    print(f"\n⚠ Corrupted images (first 5):")
    for img, err in diagnostics['images_corrupted'][:5]:
        print(f"  {img}: {err}")

if diagnostics['gt_files_empty']:
    print(f"\n⚠ Empty GT files (first 10):")
    for empty in diagnostics['gt_files_empty'][:10]:
        print(f"  {empty}")

# Calculate expected vs actual
expected_images = diagnostics['gt_files_found']
actual_images = len(images)
difference = expected_images - actual_images

print(f"\n" + "="*60)
print(f"EXPECTED VS ACTUAL:")
print(f"  Expected images: {expected_images}")
print(f"  Actual images: {actual_images}")
print(f"  Difference: {difference}")
if difference > 0:
    print(f"\n  Missing {difference} images due to:")
    print(f"    - Missing image files: {len(diagnostics['images_missing'])}")
    print(f"    - Corrupted images: {len(diagnostics['images_corrupted'])}")
    print(f"    - Empty GT files: {len(diagnostics['gt_files_empty'])}")
print("="*60)