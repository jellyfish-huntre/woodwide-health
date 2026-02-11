"""
Download and extract the PPG-DaLiA dataset from UCI Machine Learning Repository.

PPG-DaLiA is a publicly available dataset for PPG-based heart rate estimation
containing multimodal data from 15 subjects performing various physical activities.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import sys
import ssl


# Dataset information
DATASET_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
DATASET_NAME = "ppg-dalia"
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
DOWNLOAD_PATH = RAW_DATA_DIR / f"{DATASET_NAME}.zip"


def create_directories():
    """Create necessary directories for data storage."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {RAW_DATA_DIR}")


def download_dataset():
    """Download the PPG-DaLiA dataset from UCI repository."""
    if DOWNLOAD_PATH.exists():
        print(f"✓ Dataset already downloaded at {DOWNLOAD_PATH}")
        return True

    print(f"Downloading PPG-DaLiA dataset from UCI repository...")
    print(f"URL: {DATASET_URL}")
    print(f"Destination: {DOWNLOAD_PATH}")

    try:
        # Create SSL context to handle certificate verification
        # UCI repository is a trusted source
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Download with progress indication
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f} MB)")
            sys.stdout.flush()

        # Create opener with SSL context
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(DATASET_URL, DOWNLOAD_PATH, reporthook=report_progress)
        print("\n✓ Download complete!")
        return True

    except urllib.error.URLError as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def extract_dataset():
    """Extract the downloaded ZIP file."""
    if not DOWNLOAD_PATH.exists():
        print(f"✗ ZIP file not found at {DOWNLOAD_PATH}")
        return False

    extract_dir = RAW_DATA_DIR / DATASET_NAME

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"✓ Dataset already extracted at {extract_dir}")
        return True

    print(f"Extracting dataset to {extract_dir}...")

    try:
        with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("✓ Extraction complete!")
        return True

    except zipfile.BadZipFile:
        print("✗ Error: Downloaded file is not a valid ZIP file")
        return False
    except Exception as e:
        print(f"✗ Error extracting dataset: {e}")
        return False


def display_dataset_info():
    """Display information about the downloaded dataset."""
    extract_dir = RAW_DATA_DIR / DATASET_NAME

    if not extract_dir.exists():
        # Check if files were extracted directly to raw directory
        extract_dir = RAW_DATA_DIR

    print("\n" + "="*60)
    print("PPG-DaLiA Dataset Information")
    print("="*60)
    print(f"Location: {extract_dir}")

    # List contents
    if extract_dir.exists():
        files = list(extract_dir.glob("*"))
        print(f"\nFound {len(files)} items:")
        for item in sorted(files)[:10]:  # Show first 10 items
            item_type = "DIR" if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                size_mb = item.stat().st_size / (1024**2)
                size = f"({size_mb:.2f} MB)"
            print(f"  [{item_type}] {item.name} {size}")

        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more items")

    print("\nDataset Details:")
    print("  - 15 subjects performing 8 activities")
    print("  - PPG, accelerometer, and reference ECG data")
    print("  - Sampling rates: PPG 64 Hz, ACC 32 Hz, ECG 700 Hz")
    print("  - Activities: sitting, walking, cycling, etc.")
    print("="*60)


def main():
    """Main execution function."""
    print("PPG-DaLiA Dataset Download Script")
    print("="*60)

    # Step 1: Create directories
    create_directories()

    # Step 2: Download dataset
    if not download_dataset():
        print("\n✗ Failed to download dataset. Exiting.")
        return 1

    # Step 3: Extract dataset
    if not extract_dataset():
        print("\n✗ Failed to extract dataset. Exiting.")
        return 1

    # Step 4: Display information
    display_dataset_info()

    print("\n✓ Dataset ready for processing!")
    print(f"\nNext steps:")
    print(f"  1. Explore the data in: {RAW_DATA_DIR}")
    print(f"  2. Create data ingestion pipeline to load subject files")
    print(f"  3. Process time-series data for Wood Wide API")

    return 0


if __name__ == "__main__":
    sys.exit(main())
