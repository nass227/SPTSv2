import os
import json
import numpy as np
from PIL import Image

try:
    import scipy.io as sio
except ImportError:
    raise ImportError("Please install scipy: pip install scipy")


MAX_TEXT_LEN = 25
characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "


def encode_text(text, max_text_len=MAX_TEXT_LEN):
    text = text.lower()
    encoded = []

    for c in text:
        if c in CHARACTERS:
            encoded.append(CHARACTERS.index(c))
        else:
            encoded.append(0)

    encoded = encoded[:max_text_len]
    while len(encoded) < max_text_len:
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
    """
    SynthText txt field may contain:
    - nested numpy arrays
    - strings with newline-separated words/lines
    We flatten everything and split into words.
    """
    words = []

    items = np.array(txt_entry).flatten()
    for item in items:
        text = str(item).strip()
        text = text.replace("\n", " ")
        for w in text.split():
            w = w.strip()
            if w:
                words.append(w)

    return words


def parse_wordbb(wordBB):
    """
    wordBB shape is usually:
      (2, 4, N)  for N words
    or:
      (2, 4)     for single word
    Returns list of [x1, y1, x2, y2]
    """
    bb = np.array(wordBB, dtype=np.float32)

    if bb.ndim == 2:
        bb = bb[:, :, np.newaxis]
    elif bb.ndim != 3:
        return []

    boxes = []
    n_boxes = bb.shape[2]

    for k in range(n_boxes):
        xs = bb[0, :, k]
        ys = bb[1, :, k]

        x1 = float(xs.min())
        x2 = float(xs.max())
        y1 = float(ys.min())
        y2 = float(ys.max())

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return boxes


def convert_synthtext(mat_path, synth_root, output_json, max_samples=None):
    print(f"Loading mat file: {mat_path}")
    data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    imnames = data["imnames"].squeeze()
    wordBBs = data["wordBB"].squeeze()
    txts = data["txt"].squeeze()

    images = []
    annotations = []

    image_id = 1
    ann_id = 1

    total = len(imnames)
    if max_samples is not None:
        total = min(total, max_samples)

    print(f"Parsing {total} entries...")

    for i in range(total):
        img_rel = str(imnames[i][0]).strip()
        img_path = os.path.join(synth_root, img_rel)

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception as e:
            print(f"Skipping image {img_rel}: {e}")
            continue

        boxes = parse_wordbb(wordBBs[i])
        words = parse_txt_entry(txts[i])

        n = min(len(boxes), len(words))
        if n == 0:
            continue

        images.append({
            "id": image_id,
            "file_name": img_rel.replace("\\", "/"),
            "width": w,
            "height": h
        })

        for k in range(n):
            x1, y1, x2, y2 = boxes[k]
            text = words[k].strip()

            if not text:
                continue

            # clamp to image size
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = [x1, y1, x2 - x1, y2 - y1]

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "rec": encode_text(text),
                "bezier_pts": bbox_to_bezier(x1, y1, x2, y2)
            }

            annotations.append(ann)
            ann_id += 1

        image_id += 1

        if image_id % 1000 == 0:
            print(f"Processed {image_id} images...")

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "text"}]
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"Saved {output_json}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")


if __name__ == "__main__":
    SYNTH_ROOT = r"../SynthText"
    MAT_PATH = os.path.join(SYNTH_ROOT, "gt.mat")
    OUTPUT_JSON = os.path.join(SYNTH_ROOT, "train.json")

    convert_synthtext(
        mat_path=MAT_PATH,
        synth_root=SYNTH_ROOT,
        output_json=OUTPUT_JSON,
        max_samples=None
    )




    