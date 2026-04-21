import json
import re
from typing import Dict, List, Any, Tuple

CHARS: str = (
    '!"#$%&\'()*+,-./'          
    '0123456789'                 
    ':;<=>?@'                    
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
    '[\\]^_`'                    
    'abcdefghijklmnopqrstuvwxyz' 
    '{|}~'                       
    'àâäéèêëîïôùûüÿæœç'         
    'ÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ'         
)

# Create a set for O(1) lookup
VALID_CHARS = set(CHARS)
REPLACE_CHAR = '#'

def find_invalid_chars(text: str) -> List[str]:
    """Return list of unique invalid characters found in text"""
    invalid = set()
    for char in text:
        if char not in VALID_CHARS:
            invalid.add(char)
    return sorted(invalid)

def clean_annotation(ann: Dict, mode: str = 'replace', remove_triple_hash: bool = False) -> Tuple[Dict, Dict]:
    """
    Clean a single annotation
    mode: 'replace' or 'delete'
    Returns: (cleaned_annotation or None, stats)
    """
    if 'transcription' not in ann:
        return ann, {'status': 'no_transcription'}
    
    original = ann['transcription']

    if mode == 'delete' and remove_triple_hash and str(original).strip() == '###':
        return None, {
            'status': 'deleted',
            'invalid_chars': ['###'],
            'original': original[:50] + '...' if len(original) > 50 else original
        }

    invalid_chars = find_invalid_chars(original)
    
    if not invalid_chars:
        return ann, {'status': 'valid', 'invalid_chars': []}
    
    if mode == 'replace':
        # Replace invalid chars
        cleaned_text = ''.join(char if char in VALID_CHARS else REPLACE_CHAR for char in original)
        new_ann = ann.copy()
        new_ann['transcription_og'] = original
        new_ann['transcription'] = cleaned_text
        return new_ann, {
            'status': 'replaced',
            'invalid_chars': invalid_chars,
            'original': original[:50] + '...' if len(original) > 50 else original,
            'replaced': cleaned_text[:50] + '...' if len(cleaned_text) > 50 else cleaned_text
        }
    else:  # delete mode
        return None, {
            'status': 'deleted',
            'invalid_chars': invalid_chars,
            'original': original[:50] + '...' if len(original) > 50 else original
        }

def process_json(input_file: str, mode: str = 'replace', remove_triple_hash: bool = False):
    """Main processing function"""
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images...\n")
    
    cleaned_data = {}
    stats = {
        'total_annotations': 0,
        'valid_annotations': 0,
        'replaced_annotations': 0,
        'deleted_annotations': 0,
        'images_removed': 0,
        'examples': []
    }
    
    for img_id, annotations in data.items():
        if not isinstance(annotations, list):
            cleaned_data[img_id] = annotations
            continue
        
        cleaned_annotations = []
        for ann in annotations:
            stats['total_annotations'] += 1
            cleaned_ann, ann_stats = clean_annotation(ann, mode, remove_triple_hash)
            
            if ann_stats['status'] == 'valid':
                stats['valid_annotations'] += 1
                cleaned_annotations.append(cleaned_ann)
            elif ann_stats['status'] == 'replaced':
                stats['replaced_annotations'] += 1
                cleaned_annotations.append(cleaned_ann)
                if len(stats['examples']) < 10:  # Keep first 10 examples
                    stats['examples'].append({'image_id': img_id, **ann_stats})
            elif ann_stats['status'] == 'deleted':
                stats['deleted_annotations'] += 1
                if len(stats['examples']) < 10:
                    stats['examples'].append({'image_id': img_id, **ann_stats})
        
        if cleaned_annotations:
            cleaned_data[img_id] = cleaned_annotations
        else:
            stats['images_removed'] += 1
    
    return cleaned_data, stats

def main():
    input_file = r"E:\PFE\ICDAR2019\train_full_labels.json"
    output_file = input_file.replace('.json', '_cleaned.json')
    
    # print("Choose cleaning mode:")
    # print("1. Replace invalid characters with '#'")
    # print("2. Delete entire annotation if it contains invalid characters")
    # choice = input("Enter 1 or 2: ").strip()
    
    mode = 'delete'
    remove_triple_hash = True
    #  if choice == '1' else 'delete'
    
    cleaned_data, stats = process_json(input_file, mode, remove_triple_hash)
    
    print("\n" + "="*50)
    print("CLEANING STATISTICS")
    print("="*50)
    print(f"Total annotations processed: {stats['total_annotations']}")
    print(f"Valid annotations: {stats['valid_annotations']}")
    
    if mode == 'replace':
        print(f"Annotations with replaced chars: {stats['replaced_annotations']}")
        print(f"Deleted annotations: {stats['deleted_annotations']}")
    else:
        print(f"Deleted annotations (invalid chars): {stats['deleted_annotations']}")
    
    print(f"Images completely removed (empty): {stats['images_removed']}")
    print(f"Remaining images: {len(cleaned_data)}")
    
    if stats['examples']:
        print("\n" + "="*50)
        print("SAMPLE MODIFICATIONS")
        print("="*50)
        for ex in stats['examples'][:5]:
            print(f"\nImage: {ex['image_id']}")
            print(f"  Invalid chars: {ex['invalid_chars']}")
            if mode == 'replace' and 'replaced' in ex:
                print(f"  Original: {ex['original']}")
                print(f"  Replaced: {ex['replaced']}")
            else:
                print(f"  Deleted: {ex['original']}")
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()