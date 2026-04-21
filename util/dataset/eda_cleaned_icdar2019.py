import argparse
import json
from collections import Counter
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

CHARS: str = (
    '!"#$%&\'()*+,-./'
    "0123456789"
    ":;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
    "àâäéèêëîïôùûüÿæœç"
    "ÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ"
)

VALID_CHARS = set(CHARS)
REPLACE_CHAR = "#"
LATIN_LOWER = set("abcdefghijklmnopqrstuvwxyz")
LATIN_UPPER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LATIN_ALL = LATIN_LOWER | LATIN_UPPER
DIGITS = set("0123456789")
SYMBOLS = set(CHARS) - LATIN_ALL - DIGITS


def iter_annotations(data: Any) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Returns:
        - image_count
        - flat list of annotation dicts
    Supports:
        - dict[image_id] -> list[annotation]
        - COCO-like dict with an 'annotations' key
    """
    if isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
        image_count = len(data.get("images", [])) if isinstance(data.get("images"), list) else 0
        annotations = [ann for ann in data["annotations"] if isinstance(ann, dict)]
        return image_count, annotations

    if isinstance(data, dict):
        image_count = len(data)
        annotations: List[Dict[str, Any]] = []
        for anns in data.values():
            if isinstance(anns, list):
                annotations.extend([ann for ann in anns if isinstance(ann, dict)])
        return image_count, annotations

    return 0, []


def get_transcription(ann: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (field_name, transcription_value).
    Supports both transcription and rec_string naming.
    """
    if "transcription" in ann:
        return "transcription", str(ann["transcription"])
    if "rec_string" in ann:
        return "rec_string", str(ann["rec_string"])
    return "", ""


def all_chars_in_charset(text: str) -> bool:
    return all(char in VALID_CHARS for char in text)


def safe_pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def run_eda(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_count, annotations = iter_annotations(data)
    total_annotations = len(annotations)

    missing_text_key = 0
    hash_exact = 0
    contains_hash = 0
    hash_triple_exact = 0
    contains_hash_triple = 0
    clean_no_hash_in_charset = 0
    still_outside_charset = 0
    text_lengths: List[int] = []
    field_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    total_chars = 0
    latin_chars = 0
    symbol_chars = 0
    hash_chars = 0
    non_charset_chars = 0
    annotations_with_latin = 0
    annotations_with_symbols = 0
    annotations_with_non_charset = 0
    contains_werid_text = 0 
    for ann in annotations:
        field_name, text = get_transcription(ann)
        if not field_name:
            missing_text_key += 1
            continue

        field_counter[field_name] += 1
        text_lengths.append(len(text))
        total_chars += len(text)

        in_charset = all_chars_in_charset(text)
        has_hash = REPLACE_CHAR in text
        has_latin = False
        has_symbols = False
        has_non_charset = False

        for char in text:
            char_counter[char] += 1
            if char in LATIN_ALL:
                latin_chars += 1
                has_latin = True
            if char in SYMBOLS:
                symbol_chars += 1
                has_symbols = True
            if char == REPLACE_CHAR:
                hash_chars += 1
            if char not in VALID_CHARS:
                non_charset_chars += 1
                has_non_charset = True

        if text == REPLACE_CHAR:
            hash_exact += 1
        if text == "###":
            hash_triple_exact += 1
        if has_hash:
            contains_hash += 1
        if "Rääääääääääääääääääääääää" in text:
            contains_werid_text += 1
        if "###" in text:
            contains_hash_triple += 1
        if in_charset and not has_hash:
            clean_no_hash_in_charset += 1
        if not in_charset:
            still_outside_charset += 1
        if has_latin:
            annotations_with_latin += 1
        if has_symbols:
            annotations_with_symbols += 1
        if has_non_charset:
            annotations_with_non_charset += 1

    usable_annotations = total_annotations - missing_text_key

    print("=" * 70)
    print(f"Contains weird text: {contains_werid_text} "
        f"({safe_pct(contains_werid_text, usable_annotations):.2f}%)")
    print("ICDAR2019 CLEANED JSON - SMALL EDA")
    print("=" * 70)
    print(f"File: {json_path}")
    print("\n[Size]")
    print(f"Images: {image_count}")
    print(f"Annotations (total): {total_annotations}")
    print(f"Annotations with text key: {usable_annotations}")
    print(f"Annotations missing text key: {missing_text_key}")

    print("\n[Text field usage]")
    if field_counter:
        for field_name, count in field_counter.items():
            print(f"- {field_name}: {count} ({safe_pct(count, usable_annotations):.2f}%)")
    else:
        print("- No supported text field found")

    print("\n[# vs in-charset analysis]")
    print(
        f"In-charset: {clean_no_hash_in_charset} "
        f"({safe_pct(clean_no_hash_in_charset, usable_annotations):.2f}%)"
    )
    print(
        f"Contains '#': {contains_hash} "
        f"({safe_pct(contains_hash, usable_annotations):.2f}%)"
    )
    print(
        f"Exactly '#': {hash_exact} "
        f"({safe_pct(hash_exact, usable_annotations):.2f}%)"
    )
    print(
        f"Exactly '###': {hash_triple_exact} "
        f"({safe_pct(hash_triple_exact, usable_annotations):.2f}%)"
    )
    print(
        f"Contains '###': {contains_hash_triple} "
        f"({safe_pct(contains_hash_triple, usable_annotations):.2f}%)"
    )
    
    print(
        f"Still has out-of-charset chars: {still_outside_charset} "
        f"({safe_pct(still_outside_charset, usable_annotations):.2f}%)"
    )

    print("\n[Character composition]")
    print(f"Total characters: {total_chars}")
    print(
        f"Latin chars [A-Za-z]: {latin_chars} "
        f"({safe_pct(latin_chars, total_chars):.2f}%)"
    )
    print(
        f"Symbol chars (incl '#', punctuation, accents): {symbol_chars} "
        f"({safe_pct(symbol_chars, total_chars):.2f}%)"
    )
    print(
        f"'#' chars: {hash_chars} "
        f"({safe_pct(hash_chars, total_chars):.2f}%)"
    )
    print(
        f"Out-of-charset chars: {non_charset_chars} "
        f"({safe_pct(non_charset_chars, total_chars):.2f}%)"
    )

    print("\n[Annotation composition]")
    print(
        f"Annotations containing Latin: {annotations_with_latin} "
        f"({safe_pct(annotations_with_latin, usable_annotations):.2f}%)"
    )
    print(
        f"Annotations containing symbols: {annotations_with_symbols} "
        f"({safe_pct(annotations_with_symbols, usable_annotations):.2f}%)"
    )
    print(
        f"Annotations containing out-of-charset chars: {annotations_with_non_charset} "
        f"({safe_pct(annotations_with_non_charset, usable_annotations):.2f}%)"
    )

    print("\n[Top 15 most frequent characters]")
    if char_counter:
        for char, count in char_counter.most_common(15):
            label = char if char != " " else "<SPACE>"
            print(f"- {label!r}: {count} ({safe_pct(count, total_chars):.2f}%)")
    else:
        print("- No characters found.")

    print("\n[Length stats]")
    if text_lengths:
        print(f"Min length: {min(text_lengths)}")
        print(f"Median length: {median(text_lengths):.2f}")
        print(f"Mean length: {mean(text_lengths):.2f}")
        print(f"Max length: {max(text_lengths)}")
    else:
        print("No transcription/rec_string values found.")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small EDA for cleaned ICDAR2019 JSON.")
    parser.add_argument(
        "--input",
        default=r"E:\PFE\ICDAR2019\train_full_labels_cleaned.json",
        help="Path to cleaned JSON file",
    )
    args = parser.parse_args()
    run_eda(args.input)


if __name__ == "__main__":
    main()
