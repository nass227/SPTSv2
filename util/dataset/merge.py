"""
Merge COCO-style JSONs (build_annotations_json schema: bbox, area, iscrowd,
bezier_pts, rec, rec_string).  Images copied as ``{dataset}_{file_name}``;
``image_id`` / annotation ``id`` remapped globally.
ONLY includes images that have at least one valid annotation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
datasets = [
    {
        "name": "ICDAR2013",
        "ann_train": r"E:\PFE\ICDAR2013\ICDAR2013\train.json",
        "ann_test": r"E:\PFE\ICDAR2013\ICDAR2013\test.json",
        "img_train": r"E:\PFE\ICDAR2013\ICDAR2013\train_imgs",
        "img_test": r"E:\PFE\ICDAR2013\ICDAR2013\test_imgs",
    },
    {
        "name": "ICDAR2015",
        "ann_train": r"E:\PFE\ICDAR2015\ICDAR2015\train.json",
        "ann_test": r"E:\PFE\ICDAR2015\ICDAR2015\test.json",
        "img_train": r"E:\PFE\ICDAR2015\ICDAR2015\train_imgs",
        "img_test": r"E:\PFE\ICDAR2015\ICDAR2015\test_imgs",
    },
    {
        "name": "ICDAR2017",
        "ann_train": r"E:\PFE\ICDAR2017\ICDAR2017_cleaned_removed\train.json",
        "ann_test": r"E:\PFE\ICDAR2017\ICDAR2017_cleaned_removed\test.json",
        "img_train": r"E:\PFE\ICDAR2017\ICDAR2017_cleaned_removed\train_imgs",
        "img_test": r"E:\PFE\ICDAR2017\ICDAR2017_cleaned_removed\test_imgs",
    },
    {
        "name": "ICDAR2019",
        "ann_train": r"E:\PFE\ICDAR2019\split_dataset_cleaned\train.json",
        "ann_test": r"E:\PFE\ICDAR2019\split_dataset_cleaned\test.json",
        "img_train": r"E:\PFE\ICDAR2019\split_dataset_cleaned\train_images",
        "img_test": r"E:\PFE\ICDAR2019\split_dataset_cleaned\test_images",
    },
]

output_dir = r"E:\PFE\ICDAR"
output_train = os.path.join(output_dir, "images_train")
output_test = os.path.join(output_dir, "images_test")

os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

CATEGORIES = [{"id": 1, "name": "text", "supercategory": "text"}]
REQUIRED_ANN_KEYS = frozenset({"bbox", "area", "bezier_pts", "rec"})


def _load_build_charset() -> tuple[str, int]:
    path = Path(__file__).resolve().parent / "build_annotations_json.py"
    if not path.is_file():
        raise FileNotFoundError(f"merge.py needs {path} for CHARS / PAD_IDX")
    spec = importlib.util.spec_from_file_location("_build_ann_json", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CHARS, mod.PAD_IDX


CHARS, PAD_IDX = _load_build_charset()


def _decode_rec(rec: list, chars: str = CHARS, pad_idx: int = PAD_IDX) -> str:
    out: list[str] = []
    for v in rec:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i == pad_idx:
            break
        if 0 <= i < len(chars):
            out.append(chars[i])
    return "".join(out)


def _coco_schema_ok(data: dict, label: str) -> bool:
    if not isinstance(data, dict):
        print(f"[SKIP] {label} — root is not an object")
        return False
    if "images" not in data or "annotations" not in data:
        print(f"[SKIP] {label} — structure JSON incorrecte")
        return False
    if not isinstance(data["images"], list) or not isinstance(data["annotations"], list):
        print(f"[SKIP] {label} — images/annotations must be lists")
        return False
    return True


def _merge_annotation(src: dict, new_id: int, new_image_id: int) -> dict | None:
    missing = REQUIRED_ANN_KEYS - src.keys()
    if missing:
        return None
    rec = src["rec"]
    if "rec_string" in src and src["rec_string"] is not None:
        rs = str(src["rec_string"])
    else:
        rs = ""
    if not rs.strip():
        rs = _decode_rec(rec)
    return {
        "id": new_id,
        "image_id": new_image_id,
        "category_id": 1,
        "bbox": src["bbox"],
        "area": src["area"],
        "iscrowd": int(src.get("iscrowd", 0)),
        "bezier_pts": src["bezier_pts"],
        "rec": rec,
        "rec_string": rs,
    }


# ============================================================
# FONCTION DE FUSION
# ============================================================
def merge(split: str = "train") -> dict:
    merged = {
        "images": [],
        "annotations": [],
        "categories": list(CATEGORIES),
    }

    img_id = 1
    ann_id = 1
    skipped_ann = 0

    for ds in datasets:
        ann_file = ds[f"ann_{split}"]
        img_dir = ds[f"img_{split}"]

        if not os.path.exists(ann_file):
            print(f"[SKIP] {ds['name']} {split} — fichier introuvable : {ann_file}")
            continue
        if not os.path.isfile(ann_file):
            print(f"[SKIP] {ds['name']} {split} — not a file : {ann_file}")
            continue

        if os.path.getsize(ann_file) == 0:
            print(f"[SKIP] {ds['name']} {split} — fichier vide : {ann_file}")
            continue

        print(f"[INFO] Fusion {ds['name']} {split}...")

        with open(ann_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[SKIP] {ds['name']} {split} — JSON invalide : {e}")
                continue

        if not _coco_schema_ok(data, f"{ds['name']} {split}"):
            continue

        print(f"         {len(data['images'])} images, {len(data['annotations'])} annotations")

        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        skipped_no_ann = 0
        for img in data["images"]:
            old_id = img["id"]
            if old_id not in anns_by_image:
                skipped_no_ann += 1
                continue

            # Validate annotations before deciding to include this image.
            # Use a temporary image_id placeholder (img_id); real ids assigned below.
            valid_anns: list[dict] = []
            for ann in anns_by_image[old_id]:
                new_ann = _merge_annotation(ann, 0, img_id)
                if new_ann is None:
                    skipped_ann += 1
                else:
                    valid_anns.append(new_ann)

            if not valid_anns:
                # All annotations for this image failed validation — skip the image.
                skipped_no_ann += 1
                continue

            new_filename = f"{ds['name']}_{img['file_name']}"
            src = os.path.join(img_dir, img["file_name"])
            dst = os.path.join(output_train if split == "train" else output_test, new_filename)

            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            else:
                print(f"  [WARN] Image introuvable : {src}")

            merged["images"].append(
                {
                    "id": img_id,
                    "file_name": new_filename,
                    "width": int(img["width"]),
                    "height": int(img["height"]),
                }
            )

            for ann in valid_anns:
                ann["id"] = ann_id
                merged["annotations"].append(ann)
                ann_id += 1

            img_id += 1

        if skipped_no_ann:
            print(f"  [INFO] {ds['name']} {split}: {skipped_no_ann} images ignorées (aucune annotation valide)")

    if skipped_ann:
        print(
            f"  [WARN] {split}: {skipped_ann} annotations ignorées "
            f"(clés manquantes — attendu {sorted(REQUIRED_ANN_KEYS)})"
        )

    return merged


# ============================================================
# FUSION TRAIN
# ============================================================
print("=" * 50)
print("FUSION TRAIN")
print("=" * 50)
merged_train = merge("train")
out_train_ann = os.path.join(output_dir, "annotations_train_all.json")
with open(out_train_ann, "w", encoding="utf-8") as f:
    json.dump(merged_train, f, ensure_ascii=False)

print(f"\nTrain fusionné :")
print(f"  Images      : {len(merged_train['images'])}")
print(f"  Annotations : {len(merged_train['annotations'])}")

# ============================================================
# FUSION TEST
# ============================================================
print("\n" + "=" * 50)
print("FUSION TEST")
print("=" * 50)
merged_test = merge("test")
out_test_ann = os.path.join(output_dir, "annotations_test_all.json")
with open(out_test_ann, "w", encoding="utf-8") as f:
    json.dump(merged_test, f, ensure_ascii=False)

print(f"\nTest fusionné :")
print(f"  Images      : {len(merged_test['images'])}")
print(f"  Annotations : {len(merged_test['annotations'])}")

print(f"\n[DONE] Dataset fusionné dans : {output_dir}")
print(f"  Train images : {output_train}")
print(f"  Test images  : {output_test}")
print(f"  Train ann    : {out_train_ann}")
print(f"  Test ann     : {out_test_ann}")
