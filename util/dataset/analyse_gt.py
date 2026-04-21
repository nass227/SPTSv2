"""
Analyse ground-truth .txt files from one or more folders.

Supported line shapes (comma-separated; use csv quoting if the text contains commas):

1) With script / language label (ICDAR MLT-style):
   x1,y1,x2,y2,x3,y3,x4,y4,SCRIPT,TEXT
   Example: 64,166,96,166,96,179,64,179,Latin,AISHA

2) Coordinates + transcription only (ICDAR15-style):
   x1,y1,x2,y2,x3,y3,x4,y4,TEXT
   Example: 344,206,384,207,381,228,342,227,EXIT

3) Axis-aligned box + label (4 numbers, then transcription). Numbers may be separated by
   commas, spaces, tabs, semicolons, or pipes; the label may be quoted.
   Examples:
     220, 138, 598, 286, "Royal"
     46  275  539  390  London

Parsing modes (--mode):
  - auto: len(fields)==9  -> no script; len(fields)>=10 -> field 8 is script, rest is text
  - with_script: always field 8 = script, fields 9+ = text (joined)
  - coords_text: always fields 8+ = text (joined), no script column

Lines with text "###" or empty text are counted as ignored/don't-care.

Write a report file:  python ... --gt-dir DIR -o report.txt
Console only: omit -o; file only: add --quiet -o report.txt
"""
from __future__ import annotations

import argparse
import csv
import math
from io import StringIO
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

ParseMode = Literal["auto", "with_script", "coords_text"]
CoordStyle = Literal["quad8", "aabb4"]

_SEP_BETWEEN_NUMBERS = frozenset(" \t,;|")


@dataclass
class AnnRecord:
    text: str
    script: Optional[str]
    quad: list[float]
    source_file: str
    coord_style: CoordStyle = "quad8"


@dataclass
class FolderAgg:
    path: str
    records: list[AnnRecord] = field(default_factory=list)
    line_reasons: Counter[str] = field(default_factory=Counter)
    num_lines_total: int = 0
    num_difficult: int = 0        # lines whose text was "###"
    gt_files: set[str] = field(default_factory=set)


def _parse_quad(parts: list[str]) -> tuple[Optional[list[float]], Optional[str]]:
    return _parse_n_numbers(parts, 8)


def expand_if_single_csv_field(parts: list[str]) -> list[str]:
    """If the whole row was one CSV cell, re-parse inner commas (handles Excel-style one column)."""
    if len(parts) != 1:
        return parts
    blob = parts[0].strip()
    if not blob:
        return parts
    try:
        inner = next(csv.reader([blob]))
    except Exception:
        return parts
    return inner if len(inner) > 1 else parts


def _all_fields_numeric(parts: list[str]) -> bool:
    for p in parts:
        p = p.strip()
        try:
            float(p) if "." in p else float(int(p))
        except ValueError:
            return False
    return True


def quad_to_aabb(quad: list[float]) -> tuple[float, float, float, float]:
    xs = quad[0::2]
    ys = quad[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _parse_n_numbers(parts: list[str], n: int) -> tuple[Optional[list[float]], Optional[str]]:
    if len(parts) < n:
        return None, "too_few_coord_fields"
    nums: list[float] = []
    for p in parts[:n]:
        p = p.strip()
        try:
            nums.append(float(p) if "." in p else float(int(p)))
        except ValueError:
            return None, "coord_not_numeric"
    return nums, None


def aabb_xyxy_to_quad(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Two opposite corners (order-independent) -> quad TL, TR, BR, BL."""
    xa, xb = min(x1, x2), max(x1, x2)
    ya, yb = min(y1, y2), max(y1, y2)
    return [xa, ya, xb, ya, xb, yb, xa, yb]


def _read_one_number_flexible(line: str, i: int) -> tuple[Optional[float], int]:
    n = len(line)
    while i < n and line[i] in _SEP_BETWEEN_NUMBERS:
        i += 1
    if i >= n:
        return None, i
    start = i
    if line[i] in "+-":
        i += 1
    saw_digit = False
    while i < n:
        ch = line[i]
        if ch.isdigit():
            saw_digit = True
            i += 1
        elif ch == "." and "." not in line[start:i]:
            i += 1
        else:
            break
    if not saw_digit:
        return None, start
    tok = line[start:i]
    try:
        v = float(tok) if "." in tok else float(int(tok))
    except ValueError:
        return None, start
    return v, i


def parse_four_numbers_flexible(line: str) -> tuple[Optional[list[float]], str, Optional[str]]:
    """
    Read four leading numbers allowing comma / space / tab / ; / | as separators.
    The rest of the line (after separators) is the label; outer quotes are stripped.
    """
    i = 0
    nums: list[float] = []
    for _ in range(4):
        val, j = _read_one_number_flexible(line, i)
        if val is None:
            if nums:
                return None, "", "partial_numbers_then_bad_token"
            return None, "", "expected_number"
        nums.append(val)
        i = j
    while i < len(line) and line[i] in _SEP_BETWEEN_NUMBERS:
        i += 1
    text = _strip_wrapping_quotes(line[i:])
    return nums, text, None


def split_script_text(parts: list[str], mode: ParseMode) -> tuple[Optional[str], str, Optional[str]]:
    """
    Returns (script_or_none, text, error_reason).
    `parts` is the full CSV row including the first 8 coordinate fields.
    """
    if len(parts) < 9:
        return None, "", "too_few_fields"

    quad, err = _parse_quad(parts)
    if err:
        return None, "", err

    tail = parts[8:]

    if mode == "coords_text":
        text = ",".join(tail).strip()
        return None, text, None

    if mode == "with_script":
        if len(tail) < 2:
            return None, "", "with_script_requires_script_and_text"
        script = tail[0].strip()
        text = ",".join(tail[1:]).strip()
        return script or None, text, None

    # auto
    if len(parts) == 9:
        return None, tail[0].strip(), None
    script = tail[0].strip()
    text = ",".join(tail[1:]).strip()
    return script or None, text, None


def parse_gt_line(
    line: str, mode: ParseMode
) -> tuple[Optional[str], Optional[str], Optional[list[float]], Optional[str], Optional[CoordStyle]]:
    line0 = line.strip()
    if not line0:
        return None, None, None, "empty_line", None
    try:
        parts = list(next(csv.reader([line0])))
    except Exception:
        return None, None, None, "csv_parse_error", None

    parts = expand_if_single_csv_field(parts)

    # 1) Quadrilateral (8 numbers) + script/text — needs >= 9 columns after expansion
    if len(parts) >= 9:
        quad8, err = _parse_quad(parts)
        if not err:
            script, text, err2 = split_script_text(parts, mode)
            if err2:
                return None, None, quad8, err2, None
            if text == "" or text == "###":
                reason = "difficult_ignore" if text == "###" else "ignore_or_empty_text"
                return script, None, quad8, reason, None
            return script, text, quad8, None, "quad8"

    # 2) Axis-aligned box: four numbers + label (CSV columns)
    if len(parts) >= 5:
        xyxy, err = _parse_n_numbers(parts, 4)
        if not err:
            if len(parts) == 8 and _all_fields_numeric(parts):
                return None, None, None, "eight_numeric_fields_no_label", None
            text_csv = ",".join(parts[4:]).strip()
            text_csv = _strip_wrapping_quotes(text_csv)
            if text_csv == "" or text_csv == "###":
                reason = "difficult_ignore" if text_csv == "###" else "aabb_empty_text"
                return None, None, None, reason, None
            q = aabb_xyxy_to_quad(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            return None, text_csv, q, None, "aabb4"

    # 3) Same AABB layout but separators are spaces/tabs/|; or the row was one unquoted blob
    nums4, text_flex, err_f = parse_four_numbers_flexible(line0)
    if err_f:
        mapped = "unrecognized_line" if err_f == "expected_number" else err_f
        return None, None, None, mapped, None
    if text_flex == "" or text_flex == "###":
        reason = "difficult_ignore" if text_flex == "###" else "aabb_empty_text"
        return None, None, None, reason, None
    q = aabb_xyxy_to_quad(nums4[0], nums4[1], nums4[2], nums4[3])
    return None, text_flex, q, None, "aabb4"


def iter_gt_files(dirs: Iterable[Path]) -> list[tuple[Path, Path]]:
    """Return list of (folder, gt_file_path) for *.txt under each folder (non-recursive)."""
    out: list[tuple[Path, Path]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() == ".txt":
                out.append((d, p))
    return out


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def summarise_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    lengths_sorted = sorted(lengths)
    return {
        "count": len(lengths),
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": statistics.mean(lengths_sorted),
        "stdev": statistics.pstdev(lengths_sorted) if len(lengths_sorted) > 1 else 0.0,
        "p50": pct(lengths_sorted, 50),
        "p90": pct(lengths_sorted, 90),
        "p95": pct(lengths_sorted, 95),
        "p99": pct(lengths_sorted, 99),
    }


def charset_from_texts(texts: Iterable[str]) -> dict[str, Any]:
    chars: Counter[str] = Counter()
    for t in texts:
        for ch in t:
            chars[ch] += 1
    unique = sorted(chars.keys(), key=lambda c: (ord(c), c))
    return {
        "num_distinct_chars": len(unique),
        "chars_sorted": "".join(unique),
        "char_frequencies": dict(chars.most_common()),
    }


def build_report(aggs: list[FolderAgg], mode: ParseMode) -> dict[str, Any]:
    all_records: list[AnnRecord] = []
    for a in aggs:
        all_records.extend(a.records)

    texts = [r.text for r in all_records]
    lengths = [len(t) for t in texts]

    len_hist = Counter(lengths)
    script_counter: Counter[str] = Counter()
    for r in all_records:
        if r.script:
            script_counter[r.script] += 1

    bbox_w: list[float] = []
    bbox_h: list[float] = []
    aspect: list[float] = []
    for r in all_records:
        _x, _y, bw, bh = quad_to_aabb(r.quad)
        if bw > 0 and bh > 0:
            bbox_w.append(bw)
            bbox_h.append(bh)
            aspect.append(bw / bh)

    per_folder = []
    for a in aggs:
        ts = [r.text for r in a.records]
        ls = [len(t) for t in ts]
        per_folder.append(
            {
                "folder": a.path,
                "num_gt_files_scanned": len(a.gt_files),
                "num_lines": a.num_lines_total,
                "num_difficult": a.num_difficult,
                "line_reasons": dict(a.line_reasons),
                "num_valid_annotations": len(a.records),
                "length_summary": summarise_lengths(ls),
            }
        )

    report: dict[str, Any] = {
        "parse_mode": mode,
        "totals": {
            "folders": [a.path for a in aggs],
            "num_gt_files": sum(len(a.gt_files) for a in aggs),
            "num_valid_annotations": len(all_records),
            "num_difficult_ignored": sum(a.num_difficult for a in aggs),
            "num_distinct_strings": len(set(texts)),
            "coord_style_counts": dict(Counter(r.coord_style for r in all_records)),
        },
        "charset": charset_from_texts(texts),
        "text_length": summarise_lengths(lengths),
        "length_histogram_top40": len_hist.most_common(40),
        "scripts": dict(script_counter.most_common()) if script_counter else None,
        "box_stats": {
            "width_px": summarise_lengths([int(round(x)) for x in bbox_w]) if bbox_w else {},
            "height_px": summarise_lengths([int(round(x)) for x in bbox_h]) if bbox_h else {},
            "aspect_wh": summarise_lengths(aspect) if aspect else {},
        },
        "per_folder": per_folder,
        "samples": {
            "shortest": sorted(texts, key=len)[:5] if texts else [],
            "longest": sorted(texts, key=len, reverse=True)[:5] if texts else [],
        },
    }
    return report


def print_human_report(rep: dict[str, Any], stream: Any = sys.stdout) -> None:
    w = stream.write
    w("=== GT analysis ===\n")
    w(f"Parse mode: {rep['parse_mode']}\n")
    t = rep["totals"]
    w(f"Folders ({len(t['folders'])}): {', '.join(t['folders'])}\n")
    w(f"GT .txt files (sum per folder): {t['num_gt_files']}\n")
    w(f"Valid annotations: {t['num_valid_annotations']}\n")
    w(f"Difficult ('###') ignored: {t.get('num_difficult_ignored', 0)}\n")
    total_lines = sum(
        pf["num_lines"] for pf in rep.get("per_folder", [])
    )
    if total_lines:
        diff = t.get("num_difficult_ignored", 0)
        valid = t["num_valid_annotations"]
        pct_diff  = 100.0 * diff  / total_lines
        pct_valid = 100.0 * valid / total_lines
        w(f"  ({pct_valid:.1f}% readable, {pct_diff:.1f}% difficult out of {total_lines} total lines)\n")
    w(f"Distinct strings: {t['num_distinct_strings']}\n")
    if t.get("coord_style_counts"):
        w(f"Line formats: {t['coord_style_counts']}\n")
    w("\n")

    cs = rep["charset"]
    w(f"Charset: {cs['num_distinct_chars']} distinct characters\n")
    w(f"Characters (sorted): {cs['chars_sorted']}\n")
    freq = cs.get("char_frequencies") or {}
    if freq:
        w("\nCharacter counts (by frequency, then code point):\n")
        for ch, n in sorted(freq.items(), key=lambda kv: (-kv[1], ord(kv[0]) if kv[0] else 0)):
            w(f"  {ch!r}: {n}\n")
    w("\n")

    tl = rep["text_length"]
    w("Text length (characters):\n")
    for k, v in tl.items():
        w(f"  {k}: {v}\n")
    w("\nLength histogram (top 40 lengths by count):\n")
    for ln, c in rep["length_histogram_top40"]:
        w(f"  len={ln}: {c}\n")

    if rep.get("scripts"):
        w("\nScript / language column counts:\n")
        for s, c in sorted(rep["scripts"].items(), key=lambda x: -x[1]):
            w(f"  {s!r}: {c}\n")

    bs = rep["box_stats"]
    w("\nAxis-aligned box from quad (pixels, approximate):\n")
    for name, sub in bs.items():
        if sub:
            w(f"  {name}: {sub}\n")

    w("\nPer-folder summary:\n")
    for pf in rep["per_folder"]:
        w(f"  [{pf['folder']}]\n")
        w(
            f"    .txt files: {pf['num_gt_files_scanned']}, lines: {pf['num_lines']}, "
            f"valid ann: {pf['num_valid_annotations']}, difficult ('###'): {pf['num_difficult']}\n"
        )
        reasons_shown = {k: v for k, v in pf["line_reasons"].items() if k != "difficult_ignore"}
        if reasons_shown:
            w(f"    other skip reasons: {reasons_shown}\n")
        ls = pf["length_summary"]
        w(f"    length mean: {ls.get('mean', 'n/a')}, max: {ls.get('max', 'n/a')}\n")

    w("\nSample shortest / longest strings:\n")
    w(f"  shortest: {rep['samples']['shortest']}\n")
    w(f"  longest: {rep['samples']['longest']}\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse GT .txt files across folders (charset, lengths, scripts, boxes).")
    p.add_argument(
        "--gt-dir",
        action="append",
        default=[],
        help="Folder containing GT .txt files (repeat for multiple folders). Non-recursive.",
    )
    p.add_argument(
        "--mode",
        choices=("auto", "with_script", "coords_text"),
        default="auto",
        help="How to interpret CSV fields after the 8 coordinates.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        metavar="PATH",
        help="Write the full report as a UTF-8 .txt file to this path.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the report to the console (use with --output to only write the file).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    dirs = [Path(d).resolve() for d in args.gt_dir]
    if not dirs:
        print("Error: pass at least one --gt-dir folder.", file=sys.stderr)
        return 2

    mode: ParseMode = args.mode  # type: ignore[assignment]
    aggs = {str(d.resolve()): FolderAgg(path=str(d.resolve())) for d in dirs}

    for folder, gt_path in iter_gt_files(dirs):
        key = str(folder.resolve())
        agg = aggs[key]
        agg.gt_files.add(str(gt_path.resolve()))
        raw = read_text(gt_path)
        for line in raw.splitlines():
            agg.num_lines_total += 1
            script, text, quad, reason, coord_style = parse_gt_line(line, mode)
            if reason:
                if reason == "difficult_ignore":
                    agg.num_difficult += 1
                agg.line_reasons[reason] += 1
                continue
            assert text is not None and quad is not None and coord_style is not None
            agg.records.append(
                AnnRecord(
                    text=text,
                    script=script,
                    quad=quad,
                    source_file=str(gt_path),
                    coord_style=coord_style,
                )
            )

    rep = build_report(list(aggs.values()), mode)

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        print_human_report(rep, stream=buf)
        outp.write_text(buf.getvalue(), encoding="utf-8")

    if not args.quiet:
        print_human_report(rep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
