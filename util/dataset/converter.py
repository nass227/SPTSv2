import os
import json
import csv
from PIL import Image

# paths for ICDAR 2019 dataset 
IMAGE_DIR =  r"../Data/ICDAR2019/TrainImages"
GT_DIR = r"../Data/ICDAR2019/TrainGt"
OUTPUT_JSON = "../Data/ICDAR2019/train.json"

MAX_TEXT_LEN = 25
characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

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


def quad_to_bbox_and_bezier(x1, y1, x2, y2, x3, y3, x4, y4):
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]

    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    bezier_pts = [
        x1, y1,
        x2, y2,
        x2, y2,
        x3, y3,
        x3, y3,
        x4, y4,
        x4, y4,
        x1, y1
    ]

    return bbox, bezier_pts


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
        print(f"[WARN] Image not found: {img_path}")
        continue

    img = Image.open(img_path)
    w, h = img.size

    images.append({
        "id": image_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })

    with open(os.path.join(GT_DIR, file), "r", encoding="utf-8-sig") as f:

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = next(csv.reader([line]))

                if len(row) < 10:
                    print(f"[WARN] Could not parse line in {file}: {line}")
                    continue

                x1 = int(float(row[0]))
                y1 = int(float(row[1]))
                x2 = int(float(row[2]))
                y2 = int(float(row[3]))
                x3 = int(float(row[4]))
                y3 = int(float(row[5]))
                x4 = int(float(row[6]))
                y4 = int(float(row[7]))

                script = row[8].strip()
                text = ",".join(row[9:]).strip().replace('"', '')

                # keep only Latin
                if script != "Latin":
                    continue

                # skip ignored text
                if text == "###":
                    continue

                bbox, bezier_pts = quad_to_bbox_and_bezier(
                    x1, y1, x2, y2, x3, y3, x4, y4
                )

                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "rec": encode_text(text),
                    "bezier_pts": bezier_pts
                }

                annotations.append(ann)
                ann_id += 1

            except Exception:
                print(f"[WARN] Could not parse line in {file}: {line}")

    image_id += 1


coco = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 1, "name": "text"}]
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f)

print("Dataset conversion finished.")
print(f"Images: {len(images)}")
print(f"Annotations: {len(annotations)}")

