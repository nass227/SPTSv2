"""
Analyse a dataset JSON and write a UTF-8 .txt report: tree-like structure, counts,
and charset from transcription string fields.

Supports:
  - COCO-style: top-level "images" and "annotations" lists
  - Per-image / per-GT dict: many keys (e.g. "gt_1234") mapping to a list of annotation dicts

Example:
  python util/mine/analysejson.py path/to/train_full_labels.json -o report.txt
"""
from __future__ import annotations

import argparse
import json
import random
import re
from itertools import islice
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


TEXT_KEYS = ("text", "transcription", "label", "caption", "utf8_string")

LARGE_DICT_KEYS = 80
SAMPLE_KEYS_FOR_PATTERN = 400
MAX_TREE_DEPTH = 10
MAX_LIST_CHILDREN_SHOW = 3


@dataclass
class LayoutInfo:
    name: str
    num_top_level_slots: int
    num_annotations: int
    images_coco: list[Any]
    annotations_coco: list[Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse a dataset JSON and write a .txt report.")
    p.add_argument("json_path", type=str, help="Path to the JSON file.")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        metavar="PATH",
        help="Output .txt path (default: <json_stem>_analysis.txt next to the JSON file).",
    )
    return p.parse_args()


def load_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8-sig")
    return json.loads(raw)


def type_label(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def describe_scalar_preview(v: Any, max_len: int = 72) -> str:
    if isinstance(v, str):
        s = repr(v)
        return (s[: max_len - 3] + "...") if len(s) > max_len else s
    return repr(v)


def guess_key_pattern(sample_keys: list[str]) -> str | None:
    if not sample_keys:
        return None
    patterns = [
        (r"^gt_\d+$", "gt_<digits>"),
        (r"^img_\d+$", "img_<digits>"),
        (r"^\d+$", "<digits_only>"),
        (r"^image_\d+$", "image_<digits>"),
    ]
    for regex, label in patterns:
        if all(re.match(regex, k) for k in sample_keys):
            return label
    return None


def sample_dict_keys(d: dict[Any, Any], k: int) -> list[str]:
    keys = [str(x) for x in d.keys()]
    if len(keys) <= k:
        return sorted(keys)
    rng = random.Random(42)
    return sorted(rng.sample(keys, k))


def summarize_large_dict_of_values(root: dict[Any, Any], lines: list[str], indent: str) -> None:
    n = len(root)
    vals = list(root.values())
    type_counts = Counter(type(v).__name__ for v in vals)
    lines.append(f"{indent}value_type_counts (all {n} entries): {dict(type_counts)}\n")

    keys_sample = sample_dict_keys(root, min(SAMPLE_KEYS_FOR_PATTERN, n))
    pat = guess_key_pattern(keys_sample)
    if pat:
        lines.append(f"{indent}inferred_key_pattern (on {len(keys_sample)} sampled keys): {pat}\n")
    lines.append(f"{indent}sample_keys (sorted, up to 12): {sorted(keys_sample)[:12]}\n")

    if type_counts.get("list", 0) == n:
        lens = [len(v) for v in vals if isinstance(v, list)]
        if lens:
            lines.append(
                f"{indent}list lengths: min={min(lens)} max={max(lens)} "
                f"mean={statistics.mean(lens):.2f} total_items={sum(lens)}\n"
            )
        nonempty_k = next((k for k, v in root.items() if isinstance(v, list) and len(v) > 0), None)
        if nonempty_k is not None:
            first = root[nonempty_k][0]
            lines.append(f"{indent}schema_sample: first element of list under key {nonempty_k!r}\n")
            lines.extend(format_tree(first, indent + "  ", depth=1, max_depth=6, label="[0]"))


def format_tree(
    obj: Any,
    indent: str,
    depth: int,
    max_depth: int,
    label: str,
) -> list[str]:
    """Indented tree fragment (no leading root line for label '')."""
    lines: list[str] = []
    if depth > max_depth:
        lines.append(f"{indent}{label + ': ' if label else ''}...\n")
        return lines

    prefix = f"{indent}{label}: " if label else indent

    if isinstance(obj, dict):
        nk = len(obj)
        if nk > LARGE_DICT_KEYS and depth == 0:
            lines.append(f"{prefix}dict  n_keys={nk}  [large dict - summary below]\n")
            summarize_large_dict_of_values(obj, lines, indent + "  ")
            return lines
        if nk > LARGE_DICT_KEYS:
            lines.append(f"{prefix}dict  n_keys={nk}  [large nested dict — truncated]\n")
            show = sorted(str(k) for k in obj.keys())[:15]
            for sk in show:
                lines.append(f"{indent}  + {sk!r}: {type_label(obj.get(sk))}\n")
            lines.append(f"{indent}  ... ({nk - len(show)} more keys)\n")
            return lines

        lines.append(f"{prefix}dict  n_keys={nk}\n")
        keys_sorted = sorted(obj.keys(), key=lambda x: str(x))
        if nk > LARGE_DICT_KEYS and depth >= 1:
            keys_sorted = keys_sorted[:MAX_LIST_CHILDREN_SHOW]
        for k in keys_sorted:
            v = obj[k]
            sub = format_tree(v, indent + "  ", depth + 1, max_depth, str(k))
            lines.extend(sub)
        if nk > LARGE_DICT_KEYS and depth >= 1 and len(keys_sorted) < nk:
            lines.append(f"{indent}  ... ({nk - len(keys_sorted)} more keys omitted)\n")
        return lines

    if isinstance(obj, list):
        n = len(obj)
        lines.append(f"{prefix}list  length={n}\n")
        if n == 0:
            return lines
        show_n = min(MAX_LIST_CHILDREN_SHOW, n)
        for i in range(show_n):
            sub = format_tree(obj[i], indent + "  ", depth + 1, max_depth, f"[{i}]")
            lines.extend(sub)
        if n > show_n:
            lines.append(f"{indent}  ... ({n - show_n} more elements)\n")
        return lines

    if isinstance(obj, str):
        lines.append(f"{prefix}str  len={len(obj)}  {describe_scalar_preview(obj)}\n")
        return lines
    if obj is None or isinstance(obj, (bool, int, float)):
        lines.append(f"{prefix}{type_label(obj)}  {repr(obj)}\n")
        return lines

    lines.append(f"{prefix}{type_label(obj)}\n")
    return lines


def extract_coco(data: dict[Any, Any]) -> tuple[list[Any], list[Any]]:
    im = data.get("images")
    ann = data.get("annotations")
    if not isinstance(im, list):
        im = []
    if not isinstance(ann, list):
        ann = []
    return im, ann


def detect_layout(data: Any) -> LayoutInfo:
    images: list[Any] = []
    annotations_flat: list[Any] = []

    if isinstance(data, dict):
        im, ann = extract_coco(data)
        if im or ann:
            return LayoutInfo(
                name="coco_images_annotations",
                num_top_level_slots=len(im) if im else 0,
                num_annotations=len(ann),
                images_coco=im,
                annotations_coco=ann,
            )

        vals = list(data.values())
        if vals and all(isinstance(v, list) for v in vals):
            total_ann = 0
            for v in vals:
                if isinstance(v, list):
                    total_ann += len(v)
                    annotations_flat.extend(v)
            return LayoutInfo(
                name="dict_of_lists_per_key",
                num_top_level_slots=len(data),
                num_annotations=total_ann,
                images_coco=[],
                annotations_coco=annotations_flat,
            )

        for alt in ("data", "dataset"):
            sub = data.get(alt)
            if isinstance(sub, dict):
                li = detect_layout(sub)
                if li.name != "unknown":
                    return li

    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "file_name" in data[0] and "width" in data[0]:
            return LayoutInfo(
                name="root_is_images_list",
                num_top_level_slots=len(data),
                num_annotations=0,
                images_coco=data,
                annotations_coco=[],
            )
        if "image_id" in data[0] or "bbox" in data[0]:
            return LayoutInfo(
                name="root_is_annotations_list",
                num_top_level_slots=0,
                num_annotations=len(data),
                images_coco=[],
                annotations_coco=data,
            )

    return LayoutInfo("unknown", 0, 0, [], [])


def iter_annotation_dicts_for_charset(layout: LayoutInfo, data: Any) -> Iterator[dict[str, Any]]:
    if layout.name == "coco_images_annotations":
        for a in layout.annotations_coco:
            if isinstance(a, dict):
                yield a
        return
    if layout.name == "dict_of_lists_per_key" and isinstance(data, dict):
        for lst in data.values():
            if not isinstance(lst, list):
                continue
            for item in lst:
                if isinstance(item, dict):
                    yield item
        return
    if layout.name == "root_is_annotations_list":
        for a in layout.annotations_coco:
            if isinstance(a, dict):
                yield a
        return


def collect_dict_keys_from_dicts(items: Iterable[dict[str, Any]], max_items: int = 800) -> dict[str, int]:
    counts: Counter[str] = Counter()
    n = 0
    for it in items:
        if n >= max_items:
            break
        for k in it:
            counts[k] += 1
        n += 1
    return dict(counts.most_common())


def iter_annotation_text_strings(ann_iter: Iterator[dict[str, Any]]) -> Iterable[str]:
    for ann in ann_iter:
        for key in TEXT_KEYS:
            val = ann.get(key)
            if isinstance(val, str) and val.strip() and val != "###":
                yield val


def charset_report(strings: list[str]) -> tuple[str, dict[str, int], list[int]]:
    lengths = [len(s) for s in strings]
    ch_counter: Counter[str] = Counter()
    for s in strings:
        for ch in s:
            ch_counter[ch] += 1
    sorted_chars = "".join(sorted(ch_counter.keys(), key=lambda c: (ord(c), c)))
    return sorted_chars, dict(ch_counter.most_common()), lengths


def write_report(fp: TextIO, json_path: Path, data: Any, layout: LayoutInfo) -> None:
    w = fp.write
    w(f"JSON file: {json_path.resolve()}\n")
    w("=" * 72 + "\n\n")

    w("DETECTED LAYOUT\n")
    w("-" * 72 + "\n")
    w(f"layout: {layout.name}\n")
    if layout.name == "coco_images_annotations":
        w(f"  images (COCO list): {layout.num_top_level_slots}\n")
        w(f"  annotations (COCO list): {layout.num_annotations}\n")
    elif layout.name == "dict_of_lists_per_key":
        w(f"  top-level keys (logical files / images / GT ids): {layout.num_top_level_slots}\n")
        w(f"  total annotation objects (sum of all list lengths): {layout.num_annotations}\n")
    elif layout.name == "root_is_images_list":
        w(f"  root is images array: {layout.num_top_level_slots}\n")
        w(f"  annotations: {layout.num_annotations}\n")
    elif layout.name == "root_is_annotations_list":
        w(f"  root is annotations array only: {layout.num_annotations}\n")
    else:
        w("  Could not classify; see structure tree and inspect manually.\n")
    w("\n")

    w("STRUCTURE (tree)\n")
    w("-" * 72 + "\n")
    for line in format_tree(data, "", 0, MAX_TREE_DEPTH, "root"):
        w(line)
    w("\n")

    if isinstance(data, dict) and len(data) <= LARGE_DICT_KEYS:
        w("TOP-LEVEL KEYS (full list; small file)\n")
        w("-" * 72 + "\n")
        for k in sorted(data.keys(), key=str):
            v = data[k]
            extra = ""
            if isinstance(v, list):
                extra = f"  len={len(v)}"
            elif isinstance(v, dict):
                extra = f"  n_keys={len(v)}"
            w(f"  {k!r}: {type_label(v)}{extra}\n")
        w("\n")

    ann_list_head = list(islice(iter_annotation_dicts_for_charset(layout, data), 600))

    w("ANNOTATION OBJECT SCHEMA (key frequency, first up to 600 dicts scanned)\n")
    w("-" * 72 + "\n")
    if ann_list_head:
        for k, c in collect_dict_keys_from_dicts(ann_list_head).items():
            w(f"  {k!r}: {c}\n")
        has_rec = any("rec" in a for a in ann_list_head)
        has_text = any(
            any(isinstance(a.get(tk), str) and a.get(tk) for tk in TEXT_KEYS) for a in ann_list_head
        )
        if has_rec and not has_text:
            w(
                "\nNote: numeric 'rec' only — charset below uses string fields "
                f"({', '.join(TEXT_KEYS)}). Decode rec with your charset for full text stats.\n"
            )
    else:
        w("  (no annotation dicts collected for this layout)\n")
    w("\n")

    w("COUNTS SUMMARY\n")
    w("-" * 72 + "\n")
    w(f"images / top-level slots: {layout.num_top_level_slots}\n")
    w(f"annotations (total objects): {layout.num_annotations}\n\n")

    w("CHARSET (string fields on each annotation dict: " + ", ".join(TEXT_KEYS) + ")\n")
    w("-" * 72 + "\n")
    all_strings = list(iter_annotation_text_strings(iter_annotation_dicts_for_charset(layout, data)))
    if not all_strings:
        w("No non-empty string transcriptions found.\n")
    else:
        sorted_chars, freq, lengths = charset_report(all_strings)
        w(f"string values collected: {len(all_strings)}\n")
        w(f"distinct characters: {len(sorted_chars)}\n")
        w(f"characters (sorted): {sorted_chars}\n\n")
        w("character counts (by frequency, then code point):\n")
        for ch, n in sorted(freq.items(), key=lambda kv: (-kv[1], ord(kv[0]) if kv[0] else 0)):
            w(f"  {ch!r}: {n}\n")
        w("\nper-string length\n")
        w(f"  count: {len(lengths)}\n")
        w(f"  min: {min(lengths)}, max: {max(lengths)}\n")
        w(f"  mean: {statistics.mean(lengths):.4f}\n")
        if len(lengths) > 1:
            w(f"  population stdev: {statistics.pstdev(lengths):.4f}\n")
    w("\n")


def main() -> int:
    args = parse_args()
    path = Path(args.json_path)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2

    out_path = Path(args.output) if args.output else path.with_name(path.stem + "_analysis.txt")

    try:
        data = load_json(path)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Error: cannot read file: {e}", file=sys.stderr)
        return 2

    layout = detect_layout(data)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        write_report(fp, path, data, layout)

    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
