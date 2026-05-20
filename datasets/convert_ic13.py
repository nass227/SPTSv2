import os
import json
from PIL import Image

# paths for ICDAR 2013 dataset - update these paths as needed
IMAGE_DIR = r"../Data/custom/train_images"
GT_DIR = r"../Data/train_gt" 
OUTPUT_JSON = r"../Data/ICDAR2015/train.json"

MAX_TEXT_LEN = 25
characters = "0123456789abcdefghijklmnopqrstuvwxyz"

def encode_text(text):
    text = text.lower()
    encoded = []

    for c in text:
        if c in characters:
            encoded.append(characters.index(c))
        else:
            encoded.append(0)

    encoded = encoded[:MAX_TEXT_LEN]

    while len(encoded) < MAX_TEXT_LEN:
        encoded.append(0)

    return encoded


def bbox_to_bezier(x1,y1,x2,y2):
    return [
        x1,y1,
        x2,y1,
        x2,y1,
        x2,y2,
        x2,y2,
        x1,y2,
        x1,y2,
        x1,y1
    ]


images = []
annotations = []

image_id = 1
ann_id = 1

for file in os.listdir(GT_DIR):

    if not file.endswith(".txt"):
        continue

    img_name = file.replace("gt_", "").replace(".txt", ".jpg")
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    w,h = img.size

    images.append({
        "id": image_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })

    with open(os.path.join(GT_DIR,file)) as f:


        for line in f:

            parts = line.strip().replace(',', ' ').split()

            x1 = int(parts[0])
            y1 = int(parts[1])
            x2 = int(parts[2])
            y2 = int(parts[3])

            text = ",".join(parts[4:]).strip().replace('"','')

            bbox = [x1,y1,x2-x1,y2-y1]

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2]*bbox[3],
                "iscrowd": 0,
                "rec": encode_text(text),
                "bezier_pts": bbox_to_bezier(x1,y1,x2,y2)
            }

            annotations.append(ann)
            ann_id += 1

    image_id += 1


coco = {
    "images": images,
    "annotations": annotations,
    "categories":[{"id":1,"name":"text"}]
}

with open(OUTPUT_JSON,"w") as f:
    json.dump(coco,f)

print("Dataset conversion finished.")

