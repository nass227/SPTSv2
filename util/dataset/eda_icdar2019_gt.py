"""
Exploratory analysis of ICDAR2019-style ground-truth .txt files (train / test).

Expected line format (CSV): x1,y1,...,x4,y4, language_or_script, transcription
Same parsing as datasets/convert_ICDAR2019.py (csv.reader for one line).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    icdar = repo / "Data" / "ICDAR2019"
    p = argparse.ArgumentParser(description="EDA: ICDAR2019 GT language/script distribution")
    p.add_argument(
        "--train_gt",
        type=str,
        default=str(icdar / "train_gt"),
        help="Directory with training *.txt GT files (skip if missing)",
    )
    p.add_argument(
        "--test_gt",
        type=str,
        default=str(icdar / "test_gt"),
        help="Directory with test *.txt GT files (skip if missing)",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default=str(icdar / "EDA.json"),
        help="If set, write full summary JSON to this path",
    )
    return p.parse_args()


def parse_line(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (script, text, error_reason).
    error_reason is None if the line is a valid annotation with non-empty text.
    """
    line = line.strip()
    if not line:
        return None, None, "empty_line"
    try:
        parts = next(csv.reader([line]))
    except Exception:
        return None, None, "csv_parse_error"
    if len(parts) < 10:
        return None, None, "too_few_fields"
    script = parts[8].strip() if len(parts) > 8 else ""
    text = parts[9].strip() if len(parts) > 9 else ""
    if not text or text == "###":
        return script or None, None, "no_text_or_ignore"
    if not script:
        script = "(empty_script_label)"
    return script, text, None


def analyze_split(gt_dir: Path, split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "split": split_name,
        "gt_dir": str(gt_dir),
        "exists": gt_dir.is_dir(),
    }
    if not gt_dir.is_dir():
        out["note"] = "directory not found — skipped"
        return out

    gt_files = sorted(f for f in gt_dir.iterdir() if f.suffix.lower() == ".txt")
    script_counts: Counter[str] = Counter()
    char_counts: Counter[str] = Counter()
    line_stats: Counter[str] = Counter()

    for gt_path in gt_files:
        try:
            raw = gt_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = gt_path.read_text(encoding="utf-8", errors="replace")

        for line in raw.splitlines():
            script, text, err = parse_line(line)
            if err:
                line_stats[err] += 1
                continue
            assert script is not None and text is not None
            script_counts[script] += 1
            char_counts[script] += len(text)

    total_ann = sum(script_counts.values())
    out["num_gt_files"] = len(gt_files)
    out["num_valid_annotations"] = total_ann
    out["line_stats"] = dict(line_stats)
    out["scripts"] = []

    for script, n in script_counts.most_common():
        pct = (100.0 * n / total_ann) if total_ann else 0.0
        chars = char_counts[script]
        out["scripts"].append(
            {
                "script_or_language": script,
                "instances": n,
                "percent_of_annotations": round(pct, 4),
                "total_chars_in_transcriptions": chars,
                "avg_chars_per_instance": round(chars / n, 4) if n else 0.0,
            }
        )

    return out


def print_report(train: Dict[str, Any], test: Dict[str, Any]) -> None:
    def _print_split(d: Dict[str, Any]) -> None:
        name = d["split"]
        print(f"\n{'=' * 60}")
        print(f"Split: {name}")
        print(f"GT dir: {d['gt_dir']}")
        if not d.get("exists"):
            print(f"  ({d.get('note', 'missing')})")
            return
        print(f"GT files (.txt): {d['num_gt_files']}")
        print(f"Valid annotations (non-empty text, not ###): {d['num_valid_annotations']}")
        ls = d.get("line_stats") or {}
        if ls:
            print("Line-level counts (includes skipped lines):")
            for k, v in sorted(ls.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")
        rows = d.get("scripts") or []
        if not rows:
            print("No script-labeled annotations found.")
            return
        print(f"\n{'script/language':<28} {'inst':>8} {'%ann':>10} {'chars':>10} {'avg len':>10}")
        print("-" * 70)
        for r in rows:
            print(
                f"{r['script_or_language']:<28} "
                f"{r['instances']:>8} "
                f"{r['percent_of_annotations']:>9.2f}% "
                f"{r['total_chars_in_transcriptions']:>10} "
                f"{r['avg_chars_per_instance']:>10.2f}"
            )

    _print_split(train)
    _print_split(test)

    if train.get("exists") and test.get("exists"):
        ta = train.get("num_valid_annotations") or 0
        te = test.get("num_valid_annotations") or 0
        tot = ta + te
        print(f"\n{'=' * 60}")
        print("Train + test (valid annotations)")
        print(f"  Train: {ta}  |  Test: {te}  |  All: {tot}")
        if tot:
            print(f"  Train share: {100.0 * ta / tot:.2f}%  |  Test share: {100.0 * te / tot:.2f}%")


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_gt)
    test_dir = Path(args.test_gt)

    train = analyze_split(train_dir, "train")
    test = analyze_split(test_dir, "test")

    print_report(train, test)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"train": train, "test": test}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
