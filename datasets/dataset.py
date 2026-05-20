import os
import glob
import json
from collections import Counter

def analyze_dataset_character_set(gt_folder, json_folder=None):
    """
    Extract all unique characters from ICDAR 2019 dataset
    """
    all_chars = set()
    char_counter = Counter()
    
    # Read from GT txt files
    print("Reading GT files...")
    gt_files = glob.glob(f"{gt_folder}/*.txt")
    
    for gt_file in gt_files:
        with open(gt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.strip().split(',')
            if len(parts) >= 10:
                text = parts[9]  # Text is the 10th field
                for char in text:
                    if char not in ['###', 'ignore']:
                        all_chars.add(char)
                        char_counter[char] += 1
    
    # Also read from JSON if available (training annotations)
    if json_folder and os.path.exists(json_folder):
        print("Reading JSON files...")
        json_files = glob.glob(f"{json_folder}/*.json")
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'annotations' in data:
                    for ann in data['annotations']:
                        if 'rec' in ann:
                            for char in ann['rec']:
                                all_chars.add(char)
                                char_counter[char] += 1
    
    # Sort characters
    sorted_chars = sorted(all_chars)
    
    print(f"\nTotal unique characters found: {len(sorted_chars)}")
    print("\nCharacter frequency (top 50):")
    for char, count in char_counter.most_common(50):
        print(f"  '{char}' (U+{ord(char):04X}): {count} times")
    
    # Create character set string
    char_set = ''.join(sorted_chars)
    
    print(f"\nCharacter set string (length: {len(char_set)}):")
    print(char_set)
    
    # Save to file
    with open('icdar2019_charset.txt', 'w', encoding='utf-8') as f:
        f.write(char_set)
    
    print("\nSaved character set to 'icdar2019_charset.txt'")
    
    return char_set, char_counter

if __name__ == '__main__':
    # Update paths to your dataset
    gt_folder = r'C:\Users\WELTINFO\Desktop\Farida\SPTSv2\Data\ICDAR2019\train_gt'  # Your GT folder
    json_folder = r"C:\Users\WELTINFO\Desktop\Farida\Datasets\ICDAR2019\TrainGT"  # Folder containing train/test JSON files
    
    analyze_dataset_character_set(gt_folder, json_folder)