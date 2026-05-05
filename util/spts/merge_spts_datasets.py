"""
merge_spts_datasets.py
======================
Merge any number of ICDAR-spts COCO-format annotation JSONs into a single
dataset (one JSON + one image folder), handling the following automatically:

  • rec_string recovery  – when an annotation lacks rec_string (e.g. the
    ICDAR2015 spts JSONs), the text is decoded from the rec integer list
    using the 95-char printable-ASCII charset that all spts JSON files share.

  • Image renaming      – every image is copied to the output folder as
    ``{DATASET_SUFFIX}_{original_filename}`` so files from different datasets
    never collide (e.g. both IC13 and IC15 have img_1.jpg).

  • ID remapping        – image IDs and annotation IDs are reassigned starting
    from 1 so the merged JSON has no conflicts.

  • Images-only filter  – images where no annotation has
    ``annotation["image_id"] == image["id"]`` are excluded (COCO linkage only;
    ``annotation["id"]`` is unrelated and must not be confused with
    ``image["id"]``).  Invalid-only images are excluded too; the script prints why.

Usage (default config):
    python util/dataset/merge_spts_datasets.py

Usage (custom paths via CLI):
    python util/spts/merge_spts_datasets.py --split test --out_dir Data/merged_final --datasets "ic15:Data/ICDAR2015_spts/ic15_test.json:Data/ICDAR2015_spts/test_images" "ic13:Data/2013/icdar2013_spts/ic13_test.json:Data/2013/icdar2013_spts/test_images" "ic17:D:/original_icdar2017/ICDAR2017_spts/test.json:D:/original_icdar2017/ICDAR2017_spts/test_images" "ic19:E:/Nassila/ICDAR/ICDAR2019_rrc/split/test.json:E:/Nassila/ICDAR/ICDAR2019_rrc/split/test_images"



Each --datasets entry has the form  NAME:ANN_JSON:IMG_DIR

Windows ``D:/...`` paths are supported: the parser splits on the first ``:``
(for the short name) and then on ``.json:`` (case-insensitive) between the
annotation file and the image directory, so drive-letter colons are not
confused with field separators.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Charset used by all spts-style datasets
# (matches --chars default in main_original.py / main.py)
# Index = ord(char) - 32   →   ord(' ')-32 = 0, ord('~')-32 = 94
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./0123456789:;<=>?@'
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    '[\\]^_`'
    "abcdefghijklmnopqrstuvwxyz"
    '{|}~'
)          # 95 printable ASCII characters, indices 0-94
PAD_IDX: int = 96   # --pad_rec_index default (index 95 = unknown / OOV)

CATEGORIES = [
    {"id": 1, "name": "text", "supercategory": "beverage",
     "keypoints": ["mean", "xmin", "x2", "x3", "xmax",
                   "ymin", "y2", "y3", "ymax", "cross"]}
]

REQUIRED_ANN_KEYS = frozenset({"bbox", "area", "bezier_pts", "rec"})

# ---------------------------------------------------------------------------
# Default dataset configuration
# (edit these paths if you run the script without CLI arguments)
# ---------------------------------------------------------------------------
DEFAULT_SPLIT = "test"   # "train" or "test"

DEFAULT_DATASETS = [
    {
        "name": "ic15",
        "ann_test":  r"Data/ICDAR2015_spts/ic15_test.json",
        "img_test":  r"Data/ICDAR2015_spts/test_images",
        "ann_train": r"Data/ICDAR2015_spts/ic15_train.json",
        "img_train": r"Data/ICDAR2015_spts/train_images",
    },
    {
        "name": "ic13",
        "ann_test":  r"Data/2013/icdar2013_spts/ic13_test.json",
        "img_test":  r"Data/2013/icdar2013_spts/test_images",
        "ann_train": r"Data/2013/icdar2013_spts/ic13_train.json",
        "img_train": r"Data/2013/icdar2013_spts/train_images",
    },
]

DEFAULT_OUT_DIR = r"Data/merged_ic13_ic15"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_rec(rec: list, chars: str = CHARS, pad_idx: int = PAD_IDX) -> str:
    """Decode a list of charset indices back to a text string.

    Stops at the first index equal to pad_idx (or beyond the charset).
    """
    out: list[str] = []
    for v in rec:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i >= pad_idx:
            break
        if 0 <= i < len(chars):
            out.append(chars[i])
    return "".join(out)


def _validate_schema(data: dict, label: str) -> bool:
    if not isinstance(data, dict):
        print(f"[SKIP] {label} — root is not a JSON object")
        return False
    for key in ("images", "annotations"):
        if key not in data or not isinstance(data[key], list):
            print(f"[SKIP] {label} — missing or invalid '{key}' list")
            return False
    return True


def _build_annotation(
    src: dict,
    new_id: int,
    new_image_id: int,
) -> tuple[dict | None, str | None]:
    """Return ``(merged_ann, None)`` or ``(None, skip_reason)``."""
    missing = REQUIRED_ANN_KEYS - src.keys()
    if missing:
        reason = f"missing required keys {sorted(missing)}"
        return None, reason

    rec = src["rec"]

    # Prefer existing rec_string; fall back to decoding rec.
    rs: str = ""
    if "rec_string" in src and src["rec_string"] is not None:
        rs = str(src["rec_string"]).strip()
    if not rs:
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
    }, None


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def merge(
    datasets: list[dict],
    split: str,
    out_images_dir: str,
    *,
    verbose: bool = False,
) -> dict:
    """Merge all dataset splits into a single COCO-style dict.

    Parameters
    ----------
    datasets:
        List of dicts with keys: name, ann_{split}, img_{split}.
    split:
        "train" or "test".
    out_images_dir:
        Destination folder for copied (renamed) images.
    """
    os.makedirs(out_images_dir, exist_ok=True)

    merged: dict = {
        "licenses": [],
        "info": {},
        "categories": list(CATEGORIES),
        "images": [],
        "annotations": [],
    }

    global_img_id = 1
    global_ann_id = 1
    total_orphan_images = 0
    total_all_invalid_images = 0
    total_skipped_ann = 0
    total_img_missing = 0
    total_ann_skip_reasons: Counter[str] = Counter()

    for ds in datasets:
        ds_name: str = ds["name"]
        ann_key = f"ann_{split}"
        img_key = f"img_{split}"

        ann_file: str = ds.get(ann_key, "")
        img_dir: str  = ds.get(img_key, "")

        if not ann_file:
            print(f"[SKIP] {ds_name} — no annotation file configured for split '{split}'")
            continue

        if not os.path.isfile(ann_file):
            print(f"[SKIP] {ds_name} — annotation file not found: {ann_file}")
            continue

        if os.path.getsize(ann_file) == 0:
            print(f"[SKIP] {ds_name} — annotation file is empty: {ann_file}")
            continue

        print(f"\n[INFO] Merging {ds_name} ({split}) ...")
        print(f"       ann  : {ann_file}")
        print(f"       imgs : {img_dir}")

        with open(ann_file, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"[SKIP] {ds_name} — invalid JSON: {exc}")
                continue

        if not _validate_schema(data, f"{ds_name}/{split}"):
            continue

        print(f"       -> {len(data['images'])} images, {len(data['annotations'])} annotations")

        # Link annotations -> images via COCO field ``image_id`` only
        # (``ann["id"]`` is the annotation's own primary key, not the image).
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        ds_orphan_images = 0
        ds_all_invalid_images = 0
        ds_skipped_ann = 0
        ds_img_missing = 0
        ds_ann_skip_reasons: Counter[str] = Counter()
        orphan_examples: list[tuple[str, int]] = []
        all_invalid_examples: list[tuple[str, int, str]] = []
        orphan_cap = 500 if verbose else 12

        for img in data["images"]:
            old_img_id: int = img["id"]
            file_name = str(img.get("file_name", "?"))

            # No annotation has image_id == this image's id (see COCO spec).
            if old_img_id not in anns_by_image:
                ds_orphan_images += 1
                if len(orphan_examples) < orphan_cap:
                    orphan_examples.append((file_name, old_img_id))
                continue

            # Build and validate annotations for this image.
            valid_anns: list[dict] = []
            last_reason = ""
            for ann in anns_by_image[old_img_id]:
                built, skip_reason = _build_annotation(
                    ann, new_id=0, new_image_id=global_img_id
                )
                if built is None:
                    ds_skipped_ann += 1
                    last_reason = skip_reason or "unknown"
                    ds_ann_skip_reasons[skip_reason] += 1
                    total_ann_skip_reasons[skip_reason] += 1
                    if verbose:
                        print(
                            f"  [SKIP-ANN] {ds_name} image_id={old_img_id} "
                            f"ann_id={ann.get('id', '?')} :: {skip_reason}"
                        )
                else:
                    valid_anns.append(built)

            if not valid_anns:
                ds_all_invalid_images += 1
                if len(all_invalid_examples) < 8:
                    all_invalid_examples.append(
                        (file_name, old_img_id, last_reason or "unknown")
                    )
                continue

            # Rename: {dataset_name}_{original_file_name}
            original_filename: str = img["file_name"]
            stem = Path(original_filename).stem          # e.g. "img_1"
            suffix = Path(original_filename).suffix      # e.g. ".jpg"
            new_filename = f"{ds_name}_{stem}{suffix}"  # e.g. "ic15_img_1.jpg"

            # Copy image to output folder.
            src_path = os.path.join(img_dir, original_filename)
            dst_path = os.path.join(out_images_dir, new_filename)
            if os.path.isfile(src_path):
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            else:
                print(f"  [WARN] Image not found (skipping copy): {src_path}")
                ds_img_missing += 1

            # Add image entry with new global ID and renamed filename.
            merged["images"].append({
                "id":           global_img_id,
                "file_name":    new_filename,
                "width":        int(img.get("width", 0)),
                "height":       int(img.get("height", 0)),
                "coco_url":     "",
                "date_captured": "",
                "flickr_url":   "",
                "license":      0,
            })

            # Add annotation entries with new global IDs.
            for ann in valid_anns:
                ann["id"] = global_ann_id
                merged["annotations"].append(ann)
                global_ann_id += 1

            global_img_id += 1

        print(f"       [OK] kept {global_img_id - 1} images so far")

        # --- Images with zero annotations pointing at them (image.id linkage) ---
        if ds_orphan_images:
            print(
                f"       >> {ds_orphan_images} images skipped: NO_LINKED_ANNOTATIONS "
                f"(no row in 'annotations' has image_id == this image's 'id'; "
                f"do not confuse ann['id'] with ann['image_id'])"
            )
            print(
                "          COCO rule: ann['image_id'] matches image['id'] only; "
                "ann['id'] is a separate annotation id."
            )
            show_n = len(orphan_examples) if verbose else min(5, len(orphan_examples))
            for fn, iid in orphan_examples[:show_n]:
                print(f"          - file_name={fn!r}  image['id']={iid}  (no ann with image_id={iid})")
            if ds_orphan_images > show_n:
                print(f"          ... and {ds_orphan_images - show_n} more such image(s)")

        if ds_all_invalid_images:
            print(
                f"       >> {ds_all_invalid_images} images skipped: ALL_ANNOTATIONS_INVALID "
                f"(every annotation for that image failed validation)"
            )
            for fn, iid, rsn in all_invalid_examples[:5]:
                print(f"          - file_name={fn!r}  image_id={iid}  last_reason={rsn!r}")
            if verbose and len(all_invalid_examples) > 5:
                for fn, iid, rsn in all_invalid_examples[5:]:
                    print(f"          - file_name={fn!r}  image_id={iid}  last_reason={rsn!r}")

        if ds_skipped_ann:
            print(
                f"       >> {ds_skipped_ann} annotations skipped (see reasons below)"
            )
            for rsn, cnt in ds_ann_skip_reasons.most_common(8):
                print(f"          - {cnt}x  {rsn}")
            if len(ds_ann_skip_reasons) > 8:
                print(f"          ... {len(ds_ann_skip_reasons) - 8} more reason kinds")

        if ds_img_missing:
            print(f"       >> {ds_img_missing} source images not found on disk")

        total_orphan_images += ds_orphan_images
        total_all_invalid_images += ds_all_invalid_images
        total_skipped_ann += ds_skipped_ann
        total_img_missing += ds_img_missing

    print(f"\n{'='*60}")
    print(f"Merged {split}: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    if total_orphan_images:
        print(
            f"  Skipped images (no ann.image_id == image.id): {total_orphan_images}"
        )
    if total_all_invalid_images:
        print(
            f"  Skipped images (all annotations invalid)    : {total_all_invalid_images}"
        )
    if total_skipped_ann:
        print(f"  Skipped annotations (invalid)                 : {total_skipped_ann}")
        for rsn, cnt in total_ann_skip_reasons.most_common(12):
            print(f"      {cnt}x  {rsn}")
    if total_img_missing:
        print(f"  Missing source images                       : {total_img_missing}")

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_dataset_entry(entry: str) -> dict:
    """Parse a CLI dataset entry of the form  NAME:ANN_JSON:IMG_DIR.

    Cannot use ``entry.split(':', 2)`` because Windows paths contain extra
    colons (``D:/...``).  We split the short *name* on the first ``:``, then
    split *ann* from *img* at the last ``.json:`` marker (annotation file must
    end with ``.json``).  Fallback: ``rsplit(':', 1)`` for non-``.json`` ann
    paths (Unix-only).
    """
    entry = entry.strip()
    if ":" not in entry:
        raise argparse.ArgumentTypeError(
            f"Dataset entry must be NAME:ANN_JSON:IMG_DIR, got: {entry!r}"
        )
    name, rest = entry.split(":", 1)
    name = name.strip()
    rest = rest.strip()
    if not name or not rest:
        raise argparse.ArgumentTypeError(
            f"Dataset entry must be NAME:ANN_JSON:IMG_DIR, got: {entry!r}"
        )

    marker = ".json:"
    idx = rest.lower().rfind(marker)
    if idx != -1:
        ann = rest[: idx + len(".json")]
        img = rest[idx + len(marker) :].strip()
    else:
        if ":" not in rest:
            raise argparse.ArgumentTypeError(
                f"Cannot split ANN_JSON:IMG_DIR in entry (expected .json: "
                f"between paths): {entry!r}"
            )
        ann, img = rest.rsplit(":", 1)
        ann, img = ann.strip(), img.strip()

    if not ann or not img:
        raise argparse.ArgumentTypeError(
            f"Dataset entry must be NAME:ANN_JSON:IMG_DIR, got: {entry!r}"
        )

    return {"name": name, "ann_test": ann, "img_test": img,
            "ann_train": ann, "img_train": img}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge ICDAR spts-format COCO datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--split", choices=["train", "test", "both"], default=DEFAULT_SPLIT,
        help="Which split to merge (default: %(default)s).",
    )
    p.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help="Root output directory (default: %(default)s).",
    )
    p.add_argument(
        "--datasets", nargs="+", metavar="NAME:ANN:IMGDIR",
        help=(
            "Override dataset list. Each entry: NAME:ANN_JSON:IMG_DIR "
            "(Windows D:/... paths OK; ann file must end with .json). "
            "When omitted the built-in DEFAULT_DATASETS are used."
        ),
    )
    p.add_argument(
        "--chars", default=CHARS,
        help="Charset string used to decode rec lists (default: 95-char printable ASCII).",
    )
    p.add_argument(
        "--pad_idx", type=int, default=PAD_IDX,
        help="Padding index in rec lists (default: %(default)s).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print one line per skipped annotation (can be long).",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Resolve dataset list.
    if args.datasets:
        ds_list = [_parse_dataset_entry(e) for e in args.datasets]
    else:
        ds_list = DEFAULT_DATASETS

    # Resolve charset / pad (allow CLI override).
    global CHARS, PAD_IDX          # noqa: PLW0603
    CHARS   = args.chars
    PAD_IDX = args.pad_idx

    out_root = Path(args.out_dir)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  SPLIT : {split.upper()}")
        print(f"{'='*60}")

        out_images = str(out_root / f"{split}_images")
        merged = merge(ds_list, split, out_images, verbose=args.verbose)

        out_json = out_root / f"merged_{split}.json"
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, separators=(",", ":"))

        print(f"\n[DONE] JSON written to : {out_json}")
        print(f"       Images copied to : {out_images}")


if __name__ == "__main__":
    main()
