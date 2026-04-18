import os
import random
import shutil
from pathlib import Path

def split_train_test(train_imgs_folder, train_gt_folder, test_imgs_folder, test_gt_folder, train_ratio=0.8, random_seed=42):
    """
    Split ICDAR 2019 training data into training and test sets with random scrambling.
    """
    
    # Set random seed
    random.seed(random_seed)
    
    # Convert to Path objects
    train_imgs_path = Path(train_imgs_folder)
    train_gt_path = Path(train_gt_folder)
    test_imgs_path = Path(test_imgs_folder)
    test_gt_path = Path(test_gt_folder)
    
    # Verify folders exist
    if not train_imgs_path.exists():
        print(f"Error: {train_imgs_path} does not exist!")
        return
    if not train_gt_path.exists():
        print(f"Error: {train_gt_path} does not exist!")
        return
    
    # Create test folders if they don't exist
    test_imgs_path.mkdir(parents=True, exist_ok=True)
    test_gt_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files - use a SET to avoid duplicates
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    image_files = set()  # Use set to avoid duplicates
    
    for ext in image_extensions:
        # Search for both lowercase and uppercase extensions
        for file in train_imgs_path.glob(f"*{ext}"):
            image_files.add(file)
        for file in train_imgs_path.glob(f"*{ext.upper()}"):
            image_files.add(file)
    
    # Convert back to list
    image_files = list(image_files)
    
    print(f"Found {len(image_files)} unique images in {train_imgs_path}")
    
    if len(image_files) == 0:
        print("No images found! Check the path and extensions.")
        return
    
    # Match images with their ground truth files
    file_pairs = []
    missing_gt = []
    used_pairs = set()  # Track used pairs to avoid duplicates
    
    for img_path in image_files:
        # Try common GT extensions and naming conventions
        possible_gt_names = [
            img_path.stem + ".txt",
            img_path.stem + ".xml",
            img_path.stem + ".json",
            img_path.stem + ".gt.txt",
            img_path.name.replace(img_path.suffix, ".txt"),
            img_path.name.replace(img_path.suffix, ".xml"),
            img_path.stem + ".TXT",  # Uppercase variants
            img_path.stem + ".XML",
        ]
        
        gt_found = False
        for gt_name in possible_gt_names:
            gt_path = train_gt_path / gt_name
            if gt_path.exists() and (img_path, gt_path) not in used_pairs:
                file_pairs.append((img_path, gt_path))
                used_pairs.add((img_path, gt_path))
                gt_found = True
                break
        
        if not gt_found:
            missing_gt.append(img_path.name)
    
    print(f"Matched {len(file_pairs)} unique image-GT pairs")
    
    if missing_gt:
        print(f"Warning: {len(missing_gt)} images have no matching GT file")
        if len(missing_gt) <= 10:
            print("Missing GT for:", missing_gt)
    
    if len(file_pairs) == 0:
        print("No valid image-GT pairs found! Check your GT files.")
        return
    
    # Randomly shuffle the pairs (this breaks the language ordering)
    random.shuffle(file_pairs)
    
    # Calculate split index
    split_idx = int(len(file_pairs) * train_ratio)
    
    # Split the data
    train_pairs = file_pairs[:split_idx]
    test_pairs = file_pairs[split_idx:]
    
    print(f"\nSplitting data:")
    print(f"  Training set: {len(train_pairs)} pairs ({len(train_pairs)/len(file_pairs)*100:.1f}%)")
    print(f"  Test set: {len(test_pairs)} pairs ({len(test_pairs)/len(file_pairs)*100:.1f}%)")
    
    # Check if test folders already have files
    existing_test_imgs = list(test_imgs_path.glob("*"))
    existing_test_gt = list(test_gt_path.glob("*"))
    
    if existing_test_imgs or existing_test_gt:
        print(f"\nWarning: Test folders already contain:")
        print(f"  {len(existing_test_imgs)} images in test_imgs")
        print(f"  {len(existing_test_gt)} GT files in test_gt")
        response = input("Clear existing test files? (y/n): ")
        if response.lower() == 'y':
            for file in existing_test_imgs:
                try:
                    file.unlink()
                except:
                    pass
            for file in existing_test_gt:
                try:
                    file.unlink()
                except:
                    pass
            print("Cleared test folders.")
        else:
            print("Exiting to avoid conflicts.")
            return
    
    # First, verify all source files exist before moving
    print("\nVerifying source files...")
    missing_files = []
    for img_path, gt_path in test_pairs:
        if not img_path.exists():
            missing_files.append(str(img_path))
        if not gt_path.exists():
            missing_files.append(str(gt_path))
    
    if missing_files:
        print(f"Error: {len(missing_files)} source files missing!")
        print("First 10 missing files:")
        for f in missing_files[:10]:
            print(f"  {f}")
        return
    
    # Move test files
    print("\nMoving test files to test folders...")
    moved_count = 0
    failed_moves = []
    
    for img_path, gt_path in test_pairs:
        try:
            # Move image to test folder
            dest_img = test_imgs_path / img_path.name
            if img_path.exists():
                shutil.move(str(img_path), str(dest_img))
            else:
                failed_moves.append(f"Image not found: {img_path}")
                continue
            
            # Move GT to test folder
            dest_gt = test_gt_path / gt_path.name
            if gt_path.exists():
                shutil.move(str(gt_path), str(dest_gt))
            else:
                failed_moves.append(f"GT not found: {gt_path}")
                # Try to move image back if GT move fails
                if dest_img.exists():
                    shutil.move(str(dest_img), str(img_path))
                continue
                
            moved_count += 1
            
            if moved_count % 500 == 0:
                print(f"  Moved {moved_count}/{len(test_pairs)} pairs...")
                
        except Exception as e:
            failed_moves.append(f"Failed to move {img_path.name}: {str(e)}")
    
    # Verify the move was successful
    remaining_images = len([f for f in train_imgs_path.glob("*") if f.suffix.lower() in image_extensions])
    remaining_gt = len(list(train_gt_path.glob("*.txt"))) + len(list(train_gt_path.glob("*.xml")))
    
    final_test_images = len(list(test_imgs_path.glob("*")))
    final_test_gt = len(list(test_gt_path.glob("*")))
    
    print("\n" + "="*60)
    print("SPLIT COMPLETE!")
    print("="*60)
    print(f"Original training folder now contains:")
    print(f"  {remaining_images} images")
    print(f"  {remaining_gt} GT files")
    print(f"\nTest folder now contains:")
    print(f"  {final_test_images} images in {test_imgs_path}")
    print(f"  {final_test_gt} GT files in {test_gt_path}")
    
    if moved_count == len(test_pairs):
        print(f"\n✓ Successfully moved all {moved_count} test pairs!")
    else:
        print(f"\n⚠ Only moved {moved_count}/{len(test_pairs)} pairs successfully")
        if failed_moves:
            print("\nFailed moves:")
            for failure in failed_moves[:10]:
                print(f"  {failure}")
    
    # Create a summary file
    summary_file = Path.cwd() / "split_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"ICDAR 2019 Dataset Split Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Train ratio: {train_ratio}\n")
        f.write(f"Total pairs: {len(file_pairs)}\n")
        f.write(f"Training pairs: {len(train_pairs)}\n")
        f.write(f"Test pairs: {len(test_pairs)}\n")
        f.write(f"Successfully moved: {moved_count}\n")
        f.write(f"\nTest files moved:\n")
        for img_path, gt_path in test_pairs[:100]:  # List first 100
            f.write(f"  {img_path.name} -> {gt_path.name}\n")
    
    print(f"\nSummary saved to: {summary_file}")

def main():
    # ===== CONFIGURE THESE PATHS =====
    TRAIN_IMGS = r"C:/Users/L13 GEN2/Desktop/PFE/SPTSv2/Data/ICDAR2019/TrainImages/TrainImages"  # Path to your training images folder
    TRAIN_GT = r"C:/Users/L13 GEN2/Desktop/PFE/SPTSv2/Data/ICDAR2019/TrainGT/TrainGT"      # Path to your training GT folder
    TEST_IMGS = r"C:/Users/L13 GEN2/Desktop/PFE/SPTSv2/Data/ICDAR2019/test_imgs"    # Path to empty test images folder
    TEST_GT = r"C:/Users/L13 GEN2/Desktop/PFE/SPTSv2/Data/ICDAR2019/test_gt"        # Path to empty test GT folder
    TRAIN_RATIO = 0.8          # 80% training, 20% testing
    RANDOM_SEED = 42           # Change this for different randomization
    # =================================
    
    split_train_test(
        TRAIN_IMGS, 
        TRAIN_GT, 
        TEST_IMGS, 
        TEST_GT, 
        TRAIN_RATIO, 
        RANDOM_SEED
    )

if __name__ == "__main__":
    main()