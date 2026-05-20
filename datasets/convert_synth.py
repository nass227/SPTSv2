import os
import json
import random
import numpy as np
from PIL import Image
import scipy.io as sio

MAX_TEXT_LEN = 25
CHARACTERS = "0123456789abcdefghijklmnopqrstuvwxyz"

def keep_folder_range(img_rel, min_folder=1, max_folder=50):
    parts = img_rel.replace("\\", "/").split("/")
    if len(parts) < 2:
        return False

    folder_name = parts[0]
    if not folder_name.isdigit():
        return False

    folder_id = int(folder_name)
    return min_folder <= folder_id <= max_folder


def encode_text(text):
    text = text.lower()
    encoded = []

    for c in text:
        if c in CHARACTERS:
            encoded.append(CHARACTERS.index(c))
        else:
            encoded.append(0)

    encoded = encoded[:MAX_TEXT_LEN]
    while len(encoded) < MAX_TEXT_LEN:
        encoded.append(0)

    return encoded


def bbox_to_bezier(x1, y1, x2, y2):
    return [
        x1, y1,
        x2, y1,
        x2, y1,
        x2, y2,
        x2, y2,
        x1, y2,
        x1, y2,
        x1, y1
    ]


def parse_txt_entry(txt_entry):
    words = []
    items = np.array(txt_entry).flatten()
    for item in items:
        text = str(item).strip().replace("\n", " ")
        for w in text.split():
            w = w.strip()
            if w:
                words.append(w)
    return words


def parse_wordbb(wordBB):
    bb = np.array(wordBB, dtype=np.float32)

    if bb.ndim == 2:
        bb = bb[:, :, np.newaxis]
    elif bb.ndim != 3:
        return []

    boxes = []
    for k in range(bb.shape[2]):
        xs = bb[0, :, k]
        ys = bb[1, :, k]

        x1 = float(xs.min())
        x2 = float(xs.max())
        y1 = float(ys.min())
        y2 = float(ys.max())

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return boxes


def build_coco_subset(entries, synth_root, output_json):
    images = []
    annotations = []

    image_id = 1
    ann_id = 1

    for entry in entries:
        img_rel = entry["img_rel"]
        img_path = os.path.join(synth_root, img_rel)

        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            continue

        boxes = entry["boxes"]
        words = entry["words"]

        if len(boxes) == 0 or len(words) == 0:
            continue

        images.append({
            "id": image_id,
            "file_name": img_rel.replace("\\", "/"),
            "width": w,
            "height": h
        })

        n = min(len(boxes), len(words))
        for k in range(n):
            x1, y1, x2, y2 = boxes[k]
            text = words[k].strip()

            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))

            if x2 <= x1 or y2 <= y1 or not text:
                continue

            bbox = [x1, y1, x2 - x1, y2 - y1]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "rec": encode_text(text),
                "bezier_pts": bbox_to_bezier(x1, y1, x2, y2)
            })
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text"}]
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"Saved {output_json}")
    print(f"  images: {len(images)}")
    print(f"  annotations: {len(annotations)}")


def convert_synthtext_split(synth_root, mat_name="gt.mat", val_ratio=0.05, seed=42):
    mat_path = os.path.join(synth_root, mat_name)
    data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    imnames = data["imnames"].squeeze()
    wordBBs = data["wordBB"].squeeze()
    txts = data["txt"].squeeze()

    entries = []
    for i in range(len(imnames)):
        img_rel = str(imnames[i][0]).strip()
        if not keep_folder_range(img_rel, 1, 50):
            continue
        
        img_path = os.path.join(synth_root, img_rel)
        if not os.path.exists(img_path):
            continue

        boxes = parse_wordbb(wordBBs[i])
        words = parse_txt_entry(txts[i])

        if len(boxes) == 0 or len(words) == 0:
            continue

        entries.append({
            "img_rel": img_rel,
            "boxes": boxes,
            "words": words
        })

    print(f"Usable images: {len(entries)}")

    random.seed(seed)
    random.shuffle(entries)

    n_val = max(1, int(len(entries) * val_ratio))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    build_coco_subset(train_entries, synth_root, os.path.join(synth_root, "train.json"))
    build_coco_subset(val_entries, synth_root, os.path.join(synth_root, "val.json"))


if __name__ == "__main__":
    SYNTH_ROOT = r"../Datasets/SynthText"
    convert_synthtext_split(SYNTH_ROOT, mat_name="gt_filtered_1_to_100.mat", val_ratio=0.05, seed=42)
