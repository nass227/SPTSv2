import os
import json
import csv
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
TRAIN_IMAGE_DIR = r"../Data/ICDAR2019/train_imgs"
TRAIN_GT_DIR = r"../Data/ICDAR2019/train_gt"
TEST_IMAGE_DIR = r"../Data/ICDAR2019/test_imgs"
TEST_GT_DIR = r"../Data/ICDAR2019/test_gt"

OUTPUT_REPORT = "dataset_analysis_report.txt"
OUTPUT_CHARTS_DIR = "dataset_charts"

# Create charts directory
Path(OUTPUT_CHARTS_DIR).mkdir(exist_ok=True)

# ============================================
# Helper Functions
# ============================================

def get_image_info(img_path):
    """Get image dimensions and format"""
    try:
        with Image.open(img_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
    except Exception as e:
        return None

def parse_gt_file(gt_path):
    """Parse ground truth file and extract annotations"""
    annotations = []
    try:
        with open(gt_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = next(csv.reader([line]))
                    if len(parts) >= 10:
                        # Extract coordinates
                        coords = [float(p) for p in parts[:8]]
                        script = parts[8].strip() if len(parts) > 8 else "Unknown"
                        text = parts[9].strip() if len(parts) > 9 else ""
                        
                        # Calculate text length
                        text_len = len(text)
                        
                        annotations.append({
                            'script': script,
                            'text': text,
                            'text_length': text_len,
                            'coords': coords,
                            'has_text': text not in ["", "###"]
                        })
                except:
                    continue
    except Exception as e:
        pass
    return annotations

def analyze_dataset(image_dir, gt_dir, dataset_name):
    """Analyze a dataset split (train or test)"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing {dataset_name} Dataset")
    print(f"{'='*60}")
    
    # Get all GT files
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    
    # Match images with GT files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF']
    valid_pairs = []
    
    for gt_file in gt_files:
        base_name = gt_file.replace('.txt', '')
        img_found = False
        
        for ext in image_extensions:
            img_path = Path(image_dir) / (base_name + ext)
            if img_path.exists():
                valid_pairs.append({
                    'gt_file': gt_file,
                    'img_path': img_path,
                    'base_name': base_name
                })
                img_found = True
                break
        
        if not img_found:
            print(f"  Warning: No image found for {gt_file}")
    
    print(f"\nBASIC STATISTICS:")
    print(f"  Total GT files: {len(gt_files)}")
    print(f"  Valid image-GT pairs: {len(valid_pairs)}")
    
    # Analyze images
    image_stats = {
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'formats': Counter(),
        'modes': Counter(),
        'total_size_mb': 0
    }
    
    # Analyze annotations
    annotation_stats = {
        'total_annotations': 0,
        'valid_text_annotations': 0,
        'script_distribution': Counter(),
        'text_lengths': [],
        'annotations_per_image': [],
        'empty_text_count': 0
    }
    
    # Process each pair
    print(f"\nPROCESSING ANNOTATIONS...")
    
    for pair in valid_pairs:
        # Get image info
        img_info = get_image_info(pair['img_path'])
        if img_info:
            image_stats['widths'].append(img_info['width'])
            image_stats['heights'].append(img_info['height'])
            image_stats['aspect_ratios'].append(img_info['width'] / img_info['height'])
            image_stats['formats'][img_info['format']] += 1
            image_stats['modes'][img_info['mode']] += 1
            
            # Get file size
            file_size = os.path.getsize(pair['img_path']) / (1024 * 1024)  # MB
            image_stats['total_size_mb'] += file_size
        
        # Parse GT file
        gt_path = Path(gt_dir) / pair['gt_file']
        annotations = parse_gt_file(gt_path)
        
        num_annotations = len(annotations)
        annotation_stats['annotations_per_image'].append(num_annotations)
        annotation_stats['total_annotations'] += num_annotations
        
        for ann in annotations:
            if ann['has_text']:
                annotation_stats['valid_text_annotations'] += 1
                annotation_stats['script_distribution'][ann['script']] += 1
                annotation_stats['text_lengths'].append(ann['text_length'])
            else:
                annotation_stats['empty_text_count'] += 1
    
    # Calculate statistics
    print(f"\n IMAGE STATISTICS:")
    print(f"  Total images: {len(valid_pairs)}")
    print(f"  Total size: {image_stats['total_size_mb']:.2f} MB")
    print(f"  Average width: {np.mean(image_stats['widths']):.1f} px")
    print(f"  Average height: {np.mean(image_stats['heights']):.1f} px")
    print(f"  Average aspect ratio: {np.mean(image_stats['aspect_ratios']):.2f}")
    print(f"  Width range: {min(image_stats['widths'])} - {max(image_stats['widths'])} px")
    print(f"  Height range: {min(image_stats['heights'])} - {max(image_stats['heights'])} px")
    
    print(f"\n ANNOTATION STATISTICS:")
    print(f"  Total annotations: {annotation_stats['total_annotations']}")
    print(f"  Valid text annotations: {annotation_stats['valid_text_annotations']}")
    print(f"  Empty/ignored text (###): {annotation_stats['empty_text_count']}")
    print(f"  Average annotations per image: {np.mean(annotation_stats['annotations_per_image']):.2f}")
    print(f"  Min annotations per image: {min(annotation_stats['annotations_per_image'])}")
    print(f"  Max annotations per image: {max(annotation_stats['annotations_per_image'])}")
    
    print(f"\n SCRIPT/LANGUAGE DISTRIBUTION:")
    total_scripts = sum(annotation_stats['script_distribution'].values())
    for script, count in sorted(annotation_stats['script_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_scripts) * 100 if total_scripts > 0 else 0
        print(f"  {script:12s}: {count:6d} ({percentage:5.1f}%)")
    
    print(f"\n TEXT LENGTH STATISTICS:")
    if annotation_stats['text_lengths']:
        print(f"  Average text length: {np.mean(annotation_stats['text_lengths']):.2f} characters")
        print(f"  Median text length: {np.median(annotation_stats['text_lengths']):.1f} characters")
        print(f"  Min text length: {min(annotation_stats['text_lengths'])}")
        print(f"  Max text length: {max(annotation_stats['text_lengths'])}")
        
        # Text length distribution buckets
        length_buckets = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 1000)]
        print(f"\n  Text length distribution:")
        for min_len, max_len in length_buckets:
            count = sum(1 for l in annotation_stats['text_lengths'] if min_len <= l <= max_len)
            percentage = (count / len(annotation_stats['text_lengths'])) * 100
            print(f"    {min_len:3d}-{max_len:3d} chars: {count:6d} ({percentage:5.1f}%)")
    
    # Return stats for combined analysis
    return {
        'dataset_name': dataset_name,
        'num_images': len(valid_pairs),
        'num_annotations': annotation_stats['total_annotations'],
        'valid_annotations': annotation_stats['valid_text_annotations'],
        'script_distribution': annotation_stats['script_distribution'],
        'image_widths': image_stats['widths'],
        'image_heights': image_stats['heights'],
        'text_lengths': annotation_stats['text_lengths'],
        'annotations_per_image': annotation_stats['annotations_per_image']
    }

def create_visualizations(train_stats, test_stats):
    """Create charts and visualizations"""
    
    # 1. Script/Language Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Training set
    scripts_train = list(train_stats['script_distribution'].keys())
    counts_train = list(train_stats['script_distribution'].values())
    axes[0].barh(scripts_train, counts_train, color='skyblue')
    axes[0].set_xlabel('Number of Annotations')
    axes[0].set_title('Training Set - Script Distribution')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Test set
    scripts_test = list(test_stats['script_distribution'].keys())
    counts_test = list(test_stats['script_distribution'].values())
    axes[1].barh(scripts_test, counts_test, color='lightcoral')
    axes[1].set_xlabel('Number of Annotations')
    axes[1].set_title('Test Set - Script Distribution')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_CHARTS_DIR}/script_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Annotations per image histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(train_stats['annotations_per_image'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Number of Annotations per Image')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Training Set - Annotations Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(test_stats['annotations_per_image'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Number of Annotations per Image')
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title('Test Set - Annotations Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_CHARTS_DIR}/annotations_per_image.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(train_stats['text_lengths'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Text Length (characters)')
    axes[0].set_ylabel('Number of Annotations')
    axes[0].set_title('Training Set - Text Length Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].hist(test_stats['text_lengths'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Text Length (characters)')
    axes[1].set_ylabel('Number of Annotations')
    axes[1].set_title('Test Set - Text Length Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_CHARTS_DIR}/text_length_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Image dimensions scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(train_stats['image_widths'], train_stats['image_heights'], alpha=0.5, s=1, c='skyblue')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Height (pixels)')
    axes[0].set_title('Training Set - Image Dimensions')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(test_stats['image_widths'], test_stats['image_heights'], alpha=0.5, s=1, c='lightcoral')
    axes[1].set_xlabel('Width (pixels)')
    axes[1].set_ylabel('Height (pixels)')
    axes[1].set_title('Test Set - Image Dimensions')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_CHARTS_DIR}/image_dimensions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to '{OUTPUT_CHARTS_DIR}/' directory")

def generate_text_report(train_stats, test_stats):
    """Generate comprehensive text report"""
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ICDAR 2019 DATASET ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*40 + "\n")
        total_images = train_stats['num_images'] + test_stats['num_images']
        total_annotations = train_stats['num_annotations'] + test_stats['num_annotations']
        f.write(f"Total images: {total_images}\n")
        f.write(f"  - Training: {train_stats['num_images']} ({train_stats['num_images']/total_images*100:.1f}%)\n")
        f.write(f"  - Test: {test_stats['num_images']} ({test_stats['num_images']/total_images*100:.1f}%)\n")
        f.write(f"\nTotal annotations: {total_annotations}\n")
        f.write(f"  - Training: {train_stats['num_annotations']} ({train_stats['num_annotations']/total_annotations*100:.1f}%)\n")
        f.write(f"  - Test: {test_stats['num_annotations']} ({test_stats['num_annotations']/total_annotations*100:.1f}%)\n")
        
        # Per-split detailed stats
        for stats in [train_stats, test_stats]:
            f.write(f"\n{stats['dataset_name'].upper()} DATASET DETAILS\n")
            f.write("-"*40 + "\n")
            f.write(f"Images: {stats['num_images']}\n")
            f.write(f"Annotations: {stats['num_annotations']}\n")
            f.write(f"Valid text annotations: {stats['valid_annotations']}\n")
            f.write(f"Average annotations per image: {np.mean(stats['annotations_per_image']):.2f}\n")
            f.write(f"Average text length: {np.mean(stats['text_lengths']):.2f} characters\n")
            
            f.write(f"\nScript/Language Distribution:\n")
            total = sum(stats['script_distribution'].values())
            for script, count in sorted(stats['script_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                f.write(f"  {script:12s}: {count:6d} ({percentage:5.1f}%)\n")
    
    
    print(f"\nText report saved to '{OUTPUT_REPORT}'")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("\n" + "="*60)
    print("ICDAR 2019 DATASET ANALYSIS TOOL")
    print("="*60)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_IMAGE_DIR) or not os.path.exists(TRAIN_GT_DIR):
        print(f"\n Warning: Training directories not found!")
        print(f"  Images: {TRAIN_IMAGE_DIR}")
        print(f"  GT: {TRAIN_GT_DIR}")
        return
    
    if not os.path.exists(TEST_IMAGE_DIR) or not os.path.exists(TEST_GT_DIR):
        print(f"\n Warning: Test directories not found!")
        print(f"  Images: {TEST_IMAGE_DIR}")
        print(f"  GT: {TEST_GT_DIR}")
        print("\nAnalyzing training set only...")
        
        # Analyze only training set
        train_stats = analyze_dataset(TRAIN_IMAGE_DIR, TRAIN_GT_DIR, "Training")
        
        # Create dummy test stats
        test_stats = {
            'dataset_name': 'Test',
            'num_images': 0,
            'num_annotations': 0,
            'valid_annotations': 0,
            'script_distribution': {},
            'image_widths': [],
            'image_heights': [],
            'text_lengths': [],
            'annotations_per_image': []
        }
    else:
        # Analyze both splits
        train_stats = analyze_dataset(TRAIN_IMAGE_DIR, TRAIN_GT_DIR, "Training")
        test_stats = analyze_dataset(TEST_IMAGE_DIR, TEST_GT_DIR, "Test")
        
        # Create visualizations
        # print(f"\nCreating visualizations...")
        # create_visualizations(train_stats, test_stats)
    
    # Generate text report
    generate_text_report(train_stats, test_stats)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Report saved to: {OUTPUT_REPORT}")
    if os.path.exists(TEST_IMAGE_DIR):
        print(f"Charts saved to: {OUTPUT_CHARTS_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()