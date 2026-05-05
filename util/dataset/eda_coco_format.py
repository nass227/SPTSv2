import json
import numpy as np
import pandas as pd
from collections import Counter
import os

# ============================================
# CONFIGURATION
# ============================================
train_json_path = r"D:\ICDAR2019\train_annotations.json"
test_json_path = r"D:\ICDAR2019\test_annotations.json"
output_dir = r"D:\ICDAR2019\eda_results_raw.json"
os.makedirs(output_dir, exist_ok=True)

# Complete character set to check
TARGET_CHARSET = '!"#$%&\'()*+,-./0123456789:;=>ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~àâäéèêëîïôùûüÿæœçÀÂÄÉÈÊËÎÏÔÙÛÜŸÆŒÇ'

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def load_json_safe(json_path):
    """Load JSON with encoding fallback"""
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except:
            continue
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)

def analyze_split(json_path, split_name):
    """Analyze a single split and return statistics"""
    
    if not os.path.exists(json_path):
        print(f"ERROR: {split_name} file not found")
        return None
    
    data = load_json_safe(json_path)
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    if len(annotations) == 0:
        print(f"WARNING: {split_name} has no annotations")
        return None
    
    # Create dataframe
    df = pd.DataFrame(annotations)
    df['rec_string'] = df['rec_string'].fillna('').astype(str)
    df['text_length'] = df['rec_string'].str.len()
    
    # Extract all characters
    all_chars = ''.join(df['rec_string'].tolist())
    char_counts = Counter(all_chars)
    
    # Calculate charset coverage
    chars_in_dataset = set(char_counts.keys())
    target_charset_set = set(TARGET_CHARSET)
    missing_chars = target_charset_set - chars_in_dataset
    present_chars = chars_in_dataset & target_charset_set
    
    stats = {
        'split': split_name,
        'total_images': int(len(images)),
        'total_annotations': int(len(annotations)),
        'avg_texts_per_image': float(len(annotations)/len(images)) if images else 0.0,
        'unique_texts': int(df['rec_string'].nunique()),
        'hash_count': int(df['rec_string'].str.contains('#', na=False).sum()),
        'hash_percentage': float((df['rec_string'].str.contains('#', na=False).sum() / len(df)) * 100),
        'text_length_stats': {
            'min': int(df['text_length'].min()),
            'max': int(df['text_length'].max()),
            'mean': float(df['text_length'].mean()),
            'median': float(df['text_length'].median()),
            'std': float(df['text_length'].std())
        },
        'charset_stats': {
            'total_unique_chars': int(len(char_counts)),
            'target_charset_size': int(len(target_charset_set)),
            'chars_present_in_target': int(len(present_chars)),
            'chars_missing_from_target': int(len(missing_chars)),
            'chars_in_dataset': sorted(list(chars_in_dataset)),
            'missing_chars': sorted(list(missing_chars))
        },
        'top_texts': {str(k): int(v) for k, v in df['rec_string'].value_counts().head(10).to_dict().items()},
        'empty_strings': int((df['rec_string'] == '').sum()),
        'empty_percentage': float(((df['rec_string'] == '').sum() / len(df)) * 100),
        'numeric_only': int(df['rec_string'].str.isdigit().sum()),
        'alpha_only': int(df['rec_string'].str.isalpha().sum())
    }
    
    return stats

def write_report_to_file(train_stats, test_stats, output_path):
    """Write all statistics to a single text file"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("COCO DATASET EDA REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Train split
        if train_stats:
            f.write("TRAIN SPLIT ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total images: {train_stats['total_images']:,}\n")
            f.write(f"Total annotations: {train_stats['total_annotations']:,}\n")
            f.write(f"Avg texts per image: {train_stats['avg_texts_per_image']:.2f}\n")
            f.write(f"Unique text strings: {train_stats['unique_texts']:,}\n")
            f.write(f"Text length - Min: {train_stats['text_length_stats']['min']}, Max: {train_stats['text_length_stats']['max']}, Mean: {train_stats['text_length_stats']['mean']:.2f}, Median: {train_stats['text_length_stats']['median']:.0f}\n")
            f.write(f"Empty strings: {train_stats['empty_strings']:,} ({train_stats['empty_percentage']:.2f}%)\n")
            f.write(f"Numeric-only: {train_stats['numeric_only']:,} ({train_stats['numeric_only']/train_stats['total_annotations']*100:.2f}%)\n")
            f.write(f"Alpha-only: {train_stats['alpha_only']:,} ({train_stats['alpha_only']/train_stats['total_annotations']*100:.2f}%)\n")
            f.write(f"Contains '#': {train_stats['hash_count']:,} ({train_stats['hash_percentage']:.2f}%)\n\n")
            
            f.write("Top 10 most frequent texts:\n")
            for i, (text, count) in enumerate(list(train_stats['top_texts'].items())[:10], 1):
                f.write(f"  {i}. '{text}' - {count} times ({count/train_stats['total_annotations']*100:.2f}%)\n")
            f.write("\n")
            
            f.write("Character Set Analysis:\n")
            f.write(f"  Total unique characters in dataset: {train_stats['charset_stats']['total_unique_chars']}\n")
            f.write(f"  Target charset size: {train_stats['charset_stats']['target_charset_size']}\n")
            f.write(f"  Characters present in target: {train_stats['charset_stats']['chars_present_in_target']}/{train_stats['charset_stats']['target_charset_size']}\n")
            f.write(f"  Characters missing from target: {train_stats['charset_stats']['chars_missing_from_target']}\n")
            if train_stats['charset_stats']['missing_chars']:
                f.write(f"  Missing characters: {''.join(train_stats['charset_stats']['missing_chars'])}\n")
            f.write("\n")
        
        # Test split
        if test_stats:
            f.write("TEST SPLIT ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total images: {test_stats['total_images']:,}\n")
            f.write(f"Total annotations: {test_stats['total_annotations']:,}\n")
            f.write(f"Avg texts per image: {test_stats['avg_texts_per_image']:.2f}\n")
            f.write(f"Unique text strings: {test_stats['unique_texts']:,}\n")
            f.write(f"Text length - Min: {test_stats['text_length_stats']['min']}, Max: {test_stats['text_length_stats']['max']}, Mean: {test_stats['text_length_stats']['mean']:.2f}, Median: {test_stats['text_length_stats']['median']:.0f}\n")
            f.write(f"Empty strings: {test_stats['empty_strings']:,} ({test_stats['empty_percentage']:.2f}%)\n")
            f.write(f"Numeric-only: {test_stats['numeric_only']:,} ({test_stats['numeric_only']/test_stats['total_annotations']*100:.2f}%)\n")
            f.write(f"Alpha-only: {test_stats['alpha_only']:,} ({test_stats['alpha_only']/test_stats['total_annotations']*100:.2f}%)\n")
            f.write(f"Contains '#': {test_stats['hash_count']:,} ({test_stats['hash_percentage']:.2f}%)\n\n")
            
            f.write("Top 10 most frequent texts:\n")
            for i, (text, count) in enumerate(list(test_stats['top_texts'].items())[:10], 1):
                f.write(f"  {i}. '{text}' - {count} times ({count/test_stats['total_annotations']*100:.2f}%)\n")
            f.write("\n")
            
            f.write("Character Set Analysis:\n")
            f.write(f"  Total unique characters in dataset: {test_stats['charset_stats']['total_unique_chars']}\n")
            f.write(f"  Target charset size: {test_stats['charset_stats']['target_charset_size']}\n")
            f.write(f"  Characters present in target: {test_stats['charset_stats']['chars_present_in_target']}/{test_stats['charset_stats']['target_charset_size']}\n")
            f.write(f"  Characters missing from target: {test_stats['charset_stats']['chars_missing_from_target']}\n")
            if test_stats['charset_stats']['missing_chars']:
                f.write(f"  Missing characters: {''.join(test_stats['charset_stats']['missing_chars'])}\n")
            f.write("\n")
        
        # Combined dataset (train + test)
        if train_stats and test_stats:
            f.write("COMBINED DATASET (TRAIN + TEST)\n")
            f.write("-"*40 + "\n")
            
            # Load both datasets to combine
            train_data = load_json_safe(train_json_path)
            test_data = load_json_safe(test_json_path)
            
            train_anns = pd.DataFrame(train_data.get('annotations', []))
            test_anns = pd.DataFrame(test_data.get('annotations', []))
            
            combined_anns = pd.concat([train_anns, test_anns], ignore_index=True)
            combined_anns['rec_string'] = combined_anns['rec_string'].fillna('').astype(str)
            
            total_images = len(train_data.get('images', [])) + len(test_data.get('images', []))
            total_annotations = len(combined_anns)
            combined_text_length = combined_anns['rec_string'].str.len()
            combined_hash_count = combined_anns['rec_string'].str.contains('#', na=False).sum()
            combined_empty = (combined_anns['rec_string'] == '').sum()
            
            # Combined charset analysis
            combined_all_chars = ''.join(combined_anns['rec_string'].tolist())
            combined_char_counts = Counter(combined_all_chars)
            combined_chars_in_dataset = set(combined_char_counts.keys())
            target_charset_set = set(TARGET_CHARSET)
            combined_present_chars = combined_chars_in_dataset & target_charset_set
            
            f.write(f"Total images: {total_images:,}\n")
            f.write(f"Total annotations: {total_annotations:,}\n")
            f.write(f"Avg texts per image: {total_annotations/total_images:.2f}\n")
            f.write(f"Unique text strings: {combined_anns['rec_string'].nunique():,}\n")
            f.write(f"Text length - Min: {combined_text_length.min()}, Max: {combined_text_length.max()}, Mean: {combined_text_length.mean():.2f}, Median: {combined_text_length.median():.0f}\n")
            f.write(f"Empty strings: {combined_empty:,} ({combined_empty/total_annotations*100:.2f}%)\n")
            f.write(f"Contains '#': {combined_hash_count:,} ({combined_hash_count/total_annotations*100:.2f}%)\n\n")
            
            f.write("Character Set Analysis:\n")
            f.write(f"  Total unique characters in dataset: {len(combined_char_counts)}\n")
            f.write(f"  Target charset size: {len(target_charset_set)}\n")
            f.write(f"  Characters present in target: {len(combined_present_chars)}/{len(target_charset_set)}\n")
            f.write(f"  Characters missing from target: {len(target_charset_set - combined_present_chars)}\n")
            f.write("\n")
            
            # Charset listing
            f.write("CHARSET DETAILS\n")
            f.write("-"*40 + "\n")
            f.write(f"Target Charset ({len(target_charset_set)} characters):\n")
            f.write(f"{TARGET_CHARSET}\n\n")
            
            f.write(f"Characters found in dataset ({len(combined_chars_in_dataset)} total):\n")
            f.write(f"{''.join(sorted(combined_chars_in_dataset))}\n\n")
            
            missing_from_target = sorted(target_charset_set - combined_chars_in_dataset)
            if missing_from_target:
                f.write(f"Characters missing from target charset ({len(missing_from_target)}):\n")
                f.write(f"{''.join(missing_from_target)}\n\n")
            
            # Train/Test comparison
            f.write("TRAIN vs TEST COMPARISON\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Metric':<30} {'TRAIN':<15} {'TEST':<15} {'DIFF':<10}\n")
            f.write("-"*70 + "\n")
            
            metrics = [
                ('total_annotations', 'int'),
                ('unique_texts', 'int'),
                ('hash_percentage', 'pct'),
                ('empty_percentage', 'pct')
            ]
            
            for metric, mtype in metrics:
                if metric == 'total_annotations':
                    train_val = train_stats['total_annotations']
                    test_val = test_stats['total_annotations']
                elif metric == 'unique_texts':
                    train_val = train_stats['unique_texts']
                    test_val = test_stats['unique_texts']
                elif metric == 'hash_percentage':
                    train_val = train_stats['hash_percentage']
                    test_val = test_stats['hash_percentage']
                elif metric == 'empty_percentage':
                    train_val = train_stats['empty_percentage']
                    test_val = test_stats['empty_percentage']
                else:
                    continue
                
                if mtype == 'pct':
                    diff = test_val - train_val
                    f.write(f"{metric:<30} {train_val:<15.2f} {test_val:<15.2f} {diff:<+10.2f}%\n")
                elif mtype == 'int':
                    diff = test_val - train_val
                    f.write(f"{metric:<30} {train_val:<15,} {test_val:<15,} {diff:<+10,}\n")
                else:
                    diff = test_val - train_val
                    f.write(f"{metric:<30} {train_val:<15.2f} {test_val:<15.2f} {diff:<+10.2f}\n")
            
            # Data leakage check
            train_texts = set(train_anns['rec_string'].unique())
            test_texts = set(test_anns['rec_string'].unique())
            overlap = len(train_texts & test_texts)
            
            f.write(f"\nText overlap (train/test): {overlap} common strings\n")
            if overlap > 0:
                f.write(f"Overlap percentage of test: {(overlap/len(test_texts))*100:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")

def main():
    print("="*60)
    print("COCO DATASET EDA")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Analyze splits
    train_stats = analyze_split(train_json_path, 'train')
    test_stats = analyze_split(test_json_path, 'test')
    
    # Save to single text file
    output_file = os.path.join(output_dir, 'eda_report.txt')
    write_report_to_file(train_stats, test_stats, output_file)
    
    print(f"\nEDA complete! Report saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()