"""
Clean GT files by handling characters outside the charset.

Two modes (--mode):
  replace  Keep every annotation but replace out-of-charset characters with '#'.
           Output GT files are identical in structure; only the text portion changes.
           No images or annotations are removed.

  remove   Drop any annotation line whose text contains an out-of-charset character.
           If an image's GT file becomes empty after filtering, the image is also
           dropped (neither image nor GT are copied to the output).

Supported GT line formats:
  - x1,y1,x2,y2,x3,y3,x4,y4,SCRIPT,text   (ICDAR MLT-style, 10+ fields)
  - x1,y1,x2,y2,x3,y3,x4,y4,text           (8 coords + text, 9 fields)
  - x1,y1,x2,y2,text                        (AABB 4 coords + text, 5 fields)
  - Flexible separators (space / tab / ; / |) for the AABB form

Empty lines and lines with no parseable coordinates are passed through unchanged.

Usage:
  python util/mine/clean_icdar2017.py --img-dir  "E:\PFE\ICDAR2017\ch8_training_images_1" --gt-dir "E:\PFE\ICDAR2017\ch8_training_localization_transcription_gt_v2" --output   "E:\PFE\ICDAR2017\cleaned_removed"
      --mode     replace          # or: remove
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Character set — must match training config exactly
# ---------------------------------------------------------------------------
CHARS: str = (
    '!"#$%&\'()*+,-./'          # 0-15
    '0123456789'                 # 16-25
    ':;<=>?@'                    # 26-32
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # 33-58
    '[\\]^_`'                    # 59-64
    'abcdefghijklmnopqrstuvwxyz' # 65-90
    '{|}~'                       # 91-93
    'àâäéèêëîïôùûüÿæœç'         # 94-111  French lowercase
    'ÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ'         # 112-129 French uppercase
)
CHARSET_SET: frozenset[str] = frozenset(CHARS)

REPLACE_CHAR = '#'

IMAGE_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif',
    '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF', '.GIF',
]

_NUM_SEPS = frozenset(' \t,;|')


# ---------------------------------------------------------------------------
# GT line parsing — returns (prefix_fields, text) or None for unparseable lines
# ---------------------------------------------------------------------------
def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in '"\'':
        return s[1:-1]
    return s


def _parse_n_numbers(fields: list[str], n: int) -> Optional[list[float]]:
    if len(fields) < n:
        return None
    nums: list[float] = []
    for f in fields[:n]:
        try:
            nums.append(float(f.strip()))
        except ValueError:
            return None
    return nums


def _read_number_flexible(line: str, pos: int) -> tuple[Optional[float], int]:
    n = len(line)
    while pos < n and line[pos] in _NUM_SEPS:
        pos += 1
    if pos >= n:
        return None, pos
    start = pos
    if line[pos] in '+-':
        pos += 1
    saw_digit = dot_seen = False
    while pos < n:
        ch = line[pos]
        if ch.isdigit():
            saw_digit = True;  pos += 1
        elif ch == '.' and not dot_seen:
            dot_seen = True;  pos += 1
        else:
            break
    if not saw_digit:
        return None, start
    try:
        return float(line[start:pos]), pos
    except ValueError:
        return None, start


def parse_gt_line_parts(raw: str) -> Optional[tuple[list[str], str]]:
    """
    Parse a GT line into (coord_and_script_fields, text_string).
    Returns None for lines that cannot be parsed (empty / no coordinates).
    Lines that parse as coordinate-only (no text field) return None too.
    """
    line = raw.strip()
    if not line:
        return None

    try:
        fields = list(next(csv.reader([line])))
    except Exception:
        fields = [line]

    # 8-coord quad (± script label)
    if len(fields) >= 9:
        if _parse_n_numbers(fields, 8) is not None:
            tail = fields[8:]
            if (len(tail) >= 2
                    and re.match(r'^[A-Za-z][A-Za-z\-_]{0,19}$', tail[0])
                    and not tail[0].isdigit()):
                # script label present
                prefix = fields[:9]        # 8 coords + script
                text   = ','.join(tail[1:]).strip()
            else:
                prefix = fields[:8]        # 8 coords only
                text   = ','.join(tail).strip()
            text = _strip_quotes(text)
            if not text:
                return None
            return prefix, text

    # 4-coord AABB (CSV)
    if len(fields) >= 5:
        if _parse_n_numbers(fields, 4) is not None:
            prefix = fields[:4]
            text   = _strip_quotes(','.join(fields[4:]).strip())
            if not text:
                return None
            return prefix, text

    # Flexible separators — find 4 numbers then take the rest as text
    nums: list[float] = []
    pos = 0
    for _ in range(4):
        val, pos = _read_number_flexible(line, pos)
        if val is None:
            return None
        nums.append(val)
    while pos < len(line) and line[pos] in _NUM_SEPS:
        pos += 1
    text = _strip_quotes(line[pos:])
    if not text:
        return None
    # Use the raw number tokens as "prefix" (best we can do with flex format)
    prefix = [str(int(v) if v == int(v) else v) for v in nums]
    return prefix, text


def rebuild_line(prefix: list[str], text: str) -> str:
    """Write prefix fields + text back as a CSV line (no trailing newline)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='')
    writer.writerow(prefix + [text])
    return buf.getvalue()


def _has_valid_geometry(prefix: list[str]) -> bool:
    """
    Match JSON-builder geometry rules:
    - accept 4-coord (x1,y1,x2,y2) or 8-coord quad
    - bbox width and height must be strictly positive
    """
    try:
        if len(prefix) >= 8:
            nums = [float(v) for v in prefix[:8]]
            xs = nums[0::2]
            ys = nums[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        elif len(prefix) >= 4:
            nums = [float(v) for v in prefix[:4]]
            x1, x2 = min(nums[0], nums[2]), max(nums[0], nums[2])
            y1, y2 = min(nums[1], nums[3]), max(nums[1], nums[3])
        else:
            return False
    except ValueError:
        return False

    return (x2 - x1) > 0 and (y2 - y1) > 0


# ---------------------------------------------------------------------------
# Image matching
# ---------------------------------------------------------------------------
def find_image(img_dir: Path, gt_stem: str) -> Optional[Path]:
    def _exact(stem: str) -> Optional[Path]:
        for ext in IMAGE_EXTENSIONS:
            p = img_dir / f'{stem}{ext}'
            if p.is_file():
                return p
        return None

    result = _exact(gt_stem)
    if result:
        return result

    if gt_stem.startswith('gt_'):
        inner = gt_stem[3:]
        for prefix in ('', 'img_', 'image_'):
            result = _exact(prefix + inner)
            if result:
                return result

    if not gt_stem.startswith('gt_'):
        result = _exact('gt_' + gt_stem)
        if result:
            return result

    for pre in ('', 'img_', 'image_', 'gt_', 'gt_img_'):
        for suf in ('', '_img', '_image'):
            result = _exact(pre + gt_stem + suf)
            if result:
                return result

    for suf in ('_gt', '_img', '_image'):
        if gt_stem.endswith(suf):
            result = find_image(img_dir, gt_stem[: -len(suf)])
            if result:
                return result

    m = re.match(r'^\d+_(.+)$', gt_stem)
    if m:
        result = find_image(img_dir, m.group(1))
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Core cleaning logic
# ---------------------------------------------------------------------------
def clean_gt_replace(lines: list[str], stats: dict) -> list[str]:
    """Replace mode: keep every line, replacing out-of-charset chars with '#'."""
    out: list[str] = []
    for raw in lines:
        parsed = parse_gt_line_parts(raw)
        if parsed is None:
            # Unparseable / empty — pass through unchanged
            out.append(raw)
            continue

        prefix, text = parsed
        cleaned_chars: list[str] = []
        replaced = False
        for ch in text:
            if ch in CHARSET_SET:
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(REPLACE_CHAR)
                stats['total_replacements'] += 1
                stats['invalid_chars'][ch] += 1
                replaced = True

        if replaced:
            stats['annotations_with_replacements'] += 1
            cleaned_text = ''.join(cleaned_chars)
            nl = '\n' if raw.endswith('\n') else ''
            out.append(rebuild_line(prefix, cleaned_text) + nl)
        else:
            out.append(raw)

        stats['annotations_total'] += 1

    return out


def clean_gt_remove(lines: list[str], stats: dict) -> list[str]:
    """Remove mode: drop lines whose text contains out-of-charset characters."""
    out: list[str] = []
    for raw in lines:
        parsed = parse_gt_line_parts(raw)
        if parsed is None:
            # Unparseable / empty — do not carry to cleaned GT
            stats['lines_skipped_unparseable'] += 1
            continue

        prefix, text = parsed
        if not _has_valid_geometry(prefix):
            # Match downstream builder behavior (invalid bbox is skipped there)
            stats['lines_skipped_bad_geometry'] += 1
            continue

        stats['annotations_total'] += 1

        bad = {ch for ch in text if ch not in CHARSET_SET}
        if bad:
            stats['annotations_removed'] += 1
            for ch in bad:
                stats['invalid_chars'][ch] += 1
        else:
            out.append(raw)
            stats['annotations_kept'] += 1

    return out


def _pct(part: int, total: int) -> str:
    if total == 0:
        return '0.0'
    return f'{part / total * 100:.1f}'


def _repr_char(ch: str) -> str:
    specials = {'\n': '\\n', '\t': '\\t', '\r': '\\r'}
    if ch in specials:
        return specials[ch]
    # encode as \uXXXX for any char that may not display on the console
    try:
        ch.encode(sys.stdout.encoding or 'utf-8')
        return ch
    except (UnicodeEncodeError, LookupError):
        return f'\\u{ord(ch):04x}'


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(img_dir: Path, gt_dir: Path, out_dir: Path, mode: str) -> None:
    out_img_dir = out_dir / 'images'
    out_gt_dir  = out_dir / 'gt'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        'gt_files': 0,
        'images_matched': 0,
        'images_missing': [],
        # replace mode
        'annotations_total': 0,
        'annotations_with_replacements': 0,
        'total_replacements': 0,
        # remove mode
        'annotations_kept': 0,
        'annotations_removed': 0,
        'lines_skipped_unparseable': 0,
        'lines_skipped_bad_geometry': 0,
        'images_kept': 0,
        'images_dropped': 0,
        'invalid_chars': defaultdict(int),
    }

    gt_files = sorted(gt_dir.glob('*.txt'))
    stats['gt_files'] = len(gt_files)

    print(f'\nMode          : {mode}')
    print(f'Charset size  : {len(CHARS)} characters')
    print(f'GT files found: {len(gt_files)}\n')

    for gt_path in gt_files:
        img_path = find_image(img_dir, gt_path.stem)
        if img_path is None:
            stats['images_missing'].append(gt_path.name)
            continue
        stats['images_matched'] += 1

        try:
            raw_text = gt_path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            raw_text = gt_path.read_text(encoding='utf-8', errors='replace')

        lines = raw_text.splitlines(keepends=True)

        if mode == 'replace':
            out_lines = clean_gt_replace(lines, stats)
            # always copy image and write GT
            shutil.copy2(img_path, out_img_dir / img_path.name)
            (out_gt_dir / gt_path.name).write_text(
                ''.join(out_lines), encoding='utf-8'
            )
            stats['images_kept'] += 1

        else:  # remove
            out_lines = clean_gt_remove(lines, stats)
            # keep only non-empty content lines (ignore blank pass-through lines)
            has_annotation = any(
                parse_gt_line_parts(ln) is not None for ln in out_lines
            )
            if has_annotation:
                shutil.copy2(img_path, out_img_dir / img_path.name)
                (out_gt_dir / gt_path.name).write_text(
                    ''.join(out_lines), encoding='utf-8'
                )
                stats['images_kept'] += 1
            else:
                stats['images_dropped'] += 1

    # ---- summary ----
    print('=' * 60)
    print('CLEANING SUMMARY')
    print('=' * 60)
    print(f'  GT files              : {stats["gt_files"]}')
    print(f'  Images matched        : {stats["images_matched"]}')
    print(f'  Images missing        : {len(stats["images_missing"])}')

    if mode == 'replace':
        total = stats['annotations_total']
        print(f'\n  Annotations total     : {total}')
        print(f'  With replacements     : {stats["annotations_with_replacements"]}  ({_pct(stats["annotations_with_replacements"], total)}%)')
        print(f'  Characters replaced   : {stats["total_replacements"]}')
        print(f'  Images written        : {stats["images_kept"]}')
    else:
        total = stats['annotations_total']
        print(f'\n  Annotations total     : {total}')
        print(f'  Annotations kept      : {stats["annotations_kept"]}  ({_pct(stats["annotations_kept"], total)}%)')
        print(f'  Annotations removed   : {stats["annotations_removed"]}  ({_pct(stats["annotations_removed"], total)}%)')
        print(f'  Lines skipped (unparseable): {stats["lines_skipped_unparseable"]}')
        print(f'  Lines skipped (bad geometry): {stats["lines_skipped_bad_geometry"]}')
        print(f'  Images kept           : {stats["images_kept"]}')
        print(f'  Images dropped (empty GT): {stats["images_dropped"]}')

    if stats['invalid_chars']:
        label = 'replaced' if mode == 'replace' else 'caused removal'
        print(f'\n  Out-of-charset characters ({label}):')
        for ch, n in sorted(stats['invalid_chars'].items(), key=lambda x: -x[1])[:30]:
            print(f'    {_repr_char(ch)!r}: {n}')

    if stats['images_missing']:
        print(f'\n  GT files with no matching image (first 20):')
        for name in stats['images_missing'][:20]:
            print(f'    {name}')

    print(f'\n  Output images : {out_img_dir}')
    print(f'  Output GT     : {out_gt_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Clean GT files against a predefined charset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  replace  Keep all annotations; replace unknown chars with '#'.
  remove   Drop annotations with unknown chars; drop image if GT becomes empty.
""",
    )
    p.add_argument('--img-dir', required=True, help='Directory of original images.')
    p.add_argument('--gt-dir',  required=True, help='Directory of GT .txt files.')
    p.add_argument('--output',  '-o', required=True, help='Output directory (images/ and gt/ will be created inside).')
    p.add_argument('--mode', choices=('replace', 'remove'), required=True,
                   help='"replace": substitute unknown chars with "#".  "remove": drop annotation lines with unknown chars.')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    img_dir = Path(args.img_dir).resolve()
    gt_dir  = Path(args.gt_dir).resolve()
    out_dir = Path(args.output).resolve()

    if not img_dir.is_dir():
        print(f'Error: image directory not found: {img_dir}', file=sys.stderr)
        return 2
    if not gt_dir.is_dir():
        print(f'Error: GT directory not found: {gt_dir}', file=sys.stderr)
        return 2

    print(f'Image dir : {img_dir}')
    print(f'GT dir    : {gt_dir}')
    print(f'Output    : {out_dir}')

    run(img_dir, gt_dir, out_dir, args.mode)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
