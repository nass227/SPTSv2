import os
import json
import csv
from PIL import Image
import unicodedata

# paths for ICDAR 2019 dataset 
IMAGE_DIR =  r"../Data/ICDAR2019/train_imgs" 
GT_DIR = r"../Data/ICDAR2019/train_gt"
OUTPUT_JSON = "../Data/ICDAR2019/train.json"

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
    # For a quadrilateral, we can use the 4 corners and repeat them to make 8 points
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

images = []
annotations = []

image_id = 1
ann_id = 1

debug = {
    'total_lines': 0,
    'kept': 0,
    'scripts_seen': set()
}

for gt_file in os.listdir(GT_DIR):
    if not gt_file.endswith(".txt"):
        continue
    
    img_name = gt_file.replace(".txt", ".jpg")
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    if not os.path.exists(img_path):
        continue
    
    img = Image.open(img_path)
    w, h = img.size
    
    images.append({
        "id": image_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })
    
    with open(os.path.join(GT_DIR, gt_file), "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            debug['total_lines'] += 1
            
            try:
                parts = next(csv.reader([line]))
                
                if len(parts) < 10:
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
                
                script = parts[8].strip()
                text = parts[9].strip() if len(parts) > 9 else ""
                
                debug['scripts_seen'].add(script)
                
                # Keep only Latin and Symbols
                if script not in ["Latin", "Symbols"]:
                    continue
                
                if not text or text == "###":
                    continue
                
                # Convert to SPTS v2 format
                bbox, bezier_pts = quad_to_bbox_and_bezier(x1, y1, x2, y2, x3, y3, x4, y4)
                
                # Validate bbox
                if bbox[2] <= 0 or bbox[3] <= 0:
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
                ann_id += 1
                
            except Exception as e:
                continue
    
    image_id += 1

# SPTS v2 expects exactly this structure
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "text"}]
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco_format, f)

print(f"\nDone. Images: {len(images)}, Annotations: {len(annotations)}")
print(f"Total lines read: {debug['total_lines']}")
print(f"Scripts seen: {debug['scripts_seen']}")
print(f"Kept annotations: {debug['kept']}")