"""
add_rec_string_ic15.py
======================
Adds a ``rec_string`` field to every annotation in the ICDAR2015 spts JSON
files that are missing one.

The ``rec`` list uses the same 95-char printable-ASCII charset as every other
spts dataset in this project (PAD_IDX = 96).  Decoding stops at the first
index >= PAD_IDX.

Usage
-----
  # patch both files in-place (default)
  python util/dataset/add_rec_string_ic15.py

  # patch a specific file and write to a new location
  python util/dataset/add_rec_string_ic15.py \\
      --input  Data/ICDAR2015_spts/ic15_test.json \\
      --output Data/ICDAR2015_spts/ic15_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Charset — identical to main_original.py --chars default and all spts JSONs
# ---------------------------------------------------------------------------
CHARS: str = (
    ' !"#$%&\'()*+,-./'
    '0123456789'
    ':;<=>?@'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '[\\]^_`'
    'abcdefghijklmnopqrstuvwxyz'
    '{|}~'
)   # 95 chars, index = ord(char) - 32
PAD_IDX: int = 96


def decode_rec(rec: list, chars: str = CHARS, pad_idx: int = PAD_IDX) -> str:
    """Decode a rec integer list back to a text string."""
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


def patch_file(input_path: Path, output_path: Path) -> None:
    print(f"Reading  : {input_path}")
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    patched = 0
    for ann in annotations:
        if "rec_string" not in ann or not ann["rec_string"]:
            ann["rec_string"] = decode_rec(ann.get("rec", []))
            patched += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Wrote    : {output_path}")
    print(f"Patched  : {patched} annotations  "
          f"({len(annotations) - patched} already had rec_string)")


# ---------------------------------------------------------------------------
# Default file pairs (input -> output, same path = in-place)
# ---------------------------------------------------------------------------
DEFAULT_FILES = [
    ("Data/ICDAR2015_spts/ic15_test.json",  "Data/ICDAR2015_spts/ic15_test.json"),
    ("Data/ICDAR2015_spts/ic15_train.json", "Data/ICDAR2015_spts/ic15_train.json"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Add rec_string to IC15 spts JSON annotations.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", metavar="PATH",
                   help="Single input JSON to patch.")
    p.add_argument("--output", "-o", metavar="PATH",
                   help="Output path (defaults to --input, i.e. in-place).")
    args = p.parse_args(argv)

    if args.input:
        inp = Path(args.input)
        out = Path(args.output) if args.output else inp
        pairs = [(inp, out)]
    else:
        root = Path(__file__).resolve().parents[2]  # repo root
        pairs = [
            (root / src, root / dst)
            for src, dst in DEFAULT_FILES
        ]

    ok = True
    for inp, out in pairs:
        if not inp.is_file():
            print(f"[SKIP] not found: {inp}", file=sys.stderr)
            ok = False
            continue
        print()
        patch_file(inp, out)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
