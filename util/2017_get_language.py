import os
from collections import Counter

# ===== HARDCODED PATHS - CHANGE THESE =====
FOLDER_1 = r"D:\original_icdar2017\train_gt"   # Change this
FOLDER_2 = r"D:\original_icdar2017\test_gt"      # Change this
# ==========================================

def extract_data_from_gt_file(file_path):
    """Extract language labels and text lengths from a GT file."""
    languages = []
    text_lengths = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: x1,y1,x2,y2,x3,y3,x4,y4,language,text
                parts = line.split(',')
                if len(parts) >= 10:
                    language = parts[8]  # Language is the 9th field (index 8)
                    text = ','.join(parts[9:])  # Text might contain commas
                    languages.append(language)
                    text_lengths.append(len(text))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return languages, text_lengths

def analyze_folder(folder_path, folder_name):
    """Analyze all GT files in a folder."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None
    
    all_languages = []
    all_text_lengths = []
    file_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            languages, text_lengths = extract_data_from_gt_file(file_path)
            all_languages.extend(languages)
            all_text_lengths.extend(text_lengths)
            file_count += 1
    
    if file_count == 0:
        print(f"No .txt files found in {folder_path}")
        return None
    
    language_counts = Counter(all_languages)
    unique_languages = set(all_languages)
    
    # Calculate text length statistics
    if all_text_lengths:
        text_length_stats = {
            'min': min(all_text_lengths),
            'max': max(all_text_lengths),
            'mean': sum(all_text_lengths) / len(all_text_lengths),
            'total': sum(all_text_lengths),
            'all_lengths': all_text_lengths
        }
    else:
        text_length_stats = None
    
    # Per-language text length stats
    per_lang_lengths = {}
    for lang in unique_languages:
        lengths = [all_text_lengths[i] for i, l in enumerate(all_languages) if l == lang]
        if lengths:
            per_lang_lengths[lang] = {
                'min': min(lengths),
                'max': max(lengths),
                'mean': sum(lengths) / len(lengths),
                'count': len(lengths),
                'total': sum(lengths)
            }
    
    return {
        'folder_name': folder_name,
        'file_count': file_count,
        'total_annotations': len(all_languages),
        'unique_languages': unique_languages,
        'language_counts': language_counts,
        'text_length_stats': text_length_stats,
        'per_lang_lengths': per_lang_lengths,
        'all_text_lengths': all_text_lengths
    }

def print_analysis(result):
    """Pretty print the analysis results."""
    if not result:
        return
    
    print(f"\n{'='*60}")
    print(f"📁 FOLDER: {result['folder_name']}")
    print(f"{'='*60}")
    print(f"📄 Number of GT files: {result['file_count']}")
    print(f"🏷️  Total annotations (text boxes): {result['total_annotations']}")
    
    # Text length overview
    if result['text_length_stats']:
        stats = result['text_length_stats']
        print(f"\n📝 TEXT LENGTH STATISTICS (characters per text box):")
        print(f"   Min length    : {stats['min']}")
        print(f"   Max length    : {stats['max']}")
        print(f"   Mean length   : {stats['mean']:.2f}")
        print(f"   Total chars   : {stats['total']}")
    
    # Language distribution
    print(f"\n🌍 Number of unique languages: {len(result['unique_languages'])}")
    print(f"📋 Unique languages: {', '.join(sorted(result['unique_languages']))}")
    print(f"\n📊 Language distribution:")
    for lang, count in sorted(result['language_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / result['total_annotations']) * 100
        print(f"   {lang:15} : {count:5} ({percentage:5.1f}%)")
    
    # Per-language text length stats
    if result['per_lang_lengths']:
        print(f"\n📏 TEXT LENGTH BY LANGUAGE:")
        for lang, stats in sorted(result['per_lang_lengths'].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"   {lang:15} : min={stats['min']:3}, max={stats['max']:3}, "
                  f"mean={stats['mean']:6.2f}, total={stats['total']:4} chars ({stats['count']} boxes)")
    
    # Simple histogram of text lengths
    if result['all_text_lengths'] and len(result['all_text_lengths']) > 0:
        lengths = result['all_text_lengths']
        print(f"\n📊 TEXT LENGTH DISTRIBUTION (histogram):")
        # Create bins
        max_len = max(lengths)
        if max_len <= 20:
            bins = range(0, max_len + 5, 5)
        elif max_len <= 50:
            bins = range(0, max_len + 10, 10)
        else:
            bins = range(0, max_len + 20, 20)
        
        for i in range(len(bins)-1):
            bin_start = bins[i]
            bin_end = bins[i+1]
            count = sum(1 for l in lengths if bin_start <= l < bin_end)
            if count > 0:
                bar = '█' * min(count, 40)  # Cap bar length
                print(f"   {bin_start:3}-{bin_end:3}: {count:3} {bar}")
        
        # Handle the last value if max equals bin_end
        if max_len == bins[-1]:
            count = sum(1 for l in lengths if l == max_len)
            if count > 0:
                bar = '█' * min(count, 40)
                print(f"   {max_len:3}     : {count:3} {bar}")
    
    print(f"{'='*60}\n")

def main():
    print("\n" + "="*60)
    print("🔍 GROUND TRUTH EDA (Exploratory Data Analysis)")
    print("="*60)
    
    # Analyze both folders
    result1 = analyze_folder(FOLDER_1, os.path.basename(FOLDER_1))
    result2 = analyze_folder(FOLDER_2, os.path.basename(FOLDER_2))
    
    # Print results
    print_analysis(result1)
    print_analysis(result2)
    
    # Comparison summary
    if result1 and result2:
        print("\n" + "="*60)
        print("📊 COMPARISON SUMMARY")
        print("="*60)
        print(f"Languages in {result1['folder_name']} only: {result1['unique_languages'] - result2['unique_languages']}")
        print(f"Languages in {result2['folder_name']} only: {result2['unique_languages'] - result1['unique_languages']}")
        print(f"Common languages: {result1['unique_languages'] & result2['unique_languages']}")
        
        # Compare text lengths
        if result1['text_length_stats'] and result2['text_length_stats']:
            print(f"\n📏 TEXT LENGTH COMPARISON:")
            print(f"   {result1['folder_name']:20} mean: {result1['text_length_stats']['mean']:.2f} chars")
            print(f"   {result2['folder_name']:20} mean: {result2['text_length_stats']['mean']:.2f} chars")
            print(f"   Difference: {abs(result1['text_length_stats']['mean'] - result2['text_length_stats']['mean']):.2f} chars")

if __name__ == "__main__":
    main()