import os
import zipfile
import math
import shutil

def get_folder_size(folder_path):
    total_size = 0
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            total_size += os.path.getsize(full_path)
    return total_size


def check_disk_space(path, required_bytes):
    total, used, free = shutil.disk_usage(path)
    print(f"💾 Free space on target drive: {free / (1024**3):.2f} GB")
    print(f"📦 Required space (approx): {required_bytes / (1024**3):.2f} GB")

    # Add 10% safety margin
    required_with_margin = required_bytes * 1.1

    if free < required_with_margin:
        raise RuntimeError("❌ Not enough disk space to safely create ZIP files!")


def split_images_into_n_zips(folder_path, output_dir, output_prefix, num_parts):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    print("🔍 Scanning folder for images...")
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    if not images:
        print("❌ No images found.")
        return

    images.sort()
    total_images = len(images)
    chunk_size = math.ceil(total_images / num_parts)

    print(f"✅ Found {total_images} images.")

    # Calculate total folder size
    print("📊 Calculating total size...")
    total_size = get_folder_size(folder_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check disk space BEFORE starting
    check_disk_space(output_dir, total_size)

    print(f"📦 Splitting into {num_parts} ZIP files (~{chunk_size} images each)\n")

    for i in range(num_parts):
        start = i * chunk_size
        end = start + chunk_size
        chunk = images[start:end]

        zip_name = os.path.join(output_dir, f"{output_prefix}_part{i+1}.zip")
        print(f"\n🚀 Creating {zip_name} ({len(chunk)} images)...")

        try:
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
                for idx, file in enumerate(chunk, start=1):
                    full_path = os.path.join(folder_path, file)

                    try:
                        z.write(full_path, arcname=file)
                    except OSError as e:
                        print(f"\n❌ Error writing {file}: {e}")
                        return

                    # Progress update every 500 files
                    if idx % 500 == 0 or idx == len(chunk):
                        percent = (idx / len(chunk)) * 100
                        print(f"   📄 {idx}/{len(chunk)} files ({percent:.1f}%)")

        except Exception as e:
            print(f"\n❌ Failed to create {zip_name}: {e}")
            return

        print(f"✅ Finished {zip_name}")

    print("\n🎉 All ZIP files created successfully!")


# =======================
# 🔧 CONFIGURATION
# =======================
if __name__ == "__main__":
    folder = r"D:\ICDAR\images_train"   # your images
    output_dir = r"D:\ICDAR\zips"       # where ZIPs will be stored
    split_images_into_n_zips(folder, output_dir, "images", 4)