"""
Preprocess subjects from the real PPG-DaLiA dataset in data/raw/ppg-dalia.

Usage:
    python preprocess_real_dataset.py
"""

import numpy as np
from pathlib import Path
from src.ingestion.preprocess import PPGDaLiaPreprocessor


def main():
    """Preprocess all available subjects from real PPG-DaLiA dataset."""
    print("="*80)
    print("Preprocessing Real PPG-DaLiA Dataset")
    print("="*80)

    # Initialize preprocessor with real data directory
    preprocessor = PPGDaLiaPreprocessor(data_dir="data/raw")

    # Find all available subjects
    data_dir = Path("data/raw/ppg-dalia")
    available_subjects = sorted([
        int(f.stem[1:]) for f in data_dir.glob("S*.pkl")
    ])

    print(f"\nFound {len(available_subjects)} subjects: {available_subjects}")
    print()

    processed_count = 0
    failed_count = 0

    for subject_id in available_subjects:
        try:
            print(f"Processing subject {subject_id} from real dataset...")

            # Check if already processed (to avoid duplicates)
            output_file = Path(f"data/processed/subject_{subject_id:02d}_processed.pkl")
            if output_file.exists():
                print(f"  ⚠️  Subject {subject_id} already preprocessed (from synthetic data)")
                print(f"      Skipping to avoid overwrite. Delete {output_file} to reprocess.")
                continue

            processed = preprocessor.process_subject(
                subject_id=subject_id,
                window_seconds=30.0,
                stride_seconds=5.0
            )

            # Save
            preprocessor.save_processed_data(processed)

            # Display info
            print(f"  ✓ Subject {subject_id} processed successfully")
            print(f"    Windows: {processed['windows'].shape[0]}")
            print(f"    Unique activities: {np.unique(processed['labels'])}")
            print()

            processed_count += 1

        except FileNotFoundError as e:
            print(f"  ✗ Error processing subject {subject_id}: {e}")
            failed_count += 1
        except Exception as e:
            print(f"  ✗ Unexpected error for subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    print("="*80)
    print(f"Real Dataset Preprocessing Complete!")
    print(f"  ✓ Processed: {processed_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⚠️  Skipped: {len(available_subjects) - processed_count - failed_count}")
    print("="*80)

    # Show all processed subjects
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        processed_files = sorted(processed_dir.glob("subject_*_processed.pkl"))
        print(f"\nTotal processed subjects: {len(processed_files)}")
        for f in processed_files:
            size_mb = f.stat().st_size / (1024**2)
            print(f"  - {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
