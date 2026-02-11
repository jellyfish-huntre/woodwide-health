"""
Batch processing script to preprocess all subjects in the PPG-DaLiA dataset.

This script processes all 15 subjects and creates windowed feature vectors
ready for embedding generation via the Wood Wide API.
"""

import argparse
from pathlib import Path
from src.ingestion.preprocess import PPGDaLiaPreprocessor
import sys


def process_all_subjects(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    window_seconds: float = 30.0,
    stride_seconds: float = 5.0,
    subjects: list = None
):
    """
    Process all subjects in the dataset.

    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data
        window_seconds: Window size for feature vectors
        stride_seconds: Stride between windows
        subjects: List of subject IDs to process (default: 1-15)
    """
    if subjects is None:
        subjects = range(1, 16)  # 15 subjects in PPG-DaLiA

    preprocessor = PPGDaLiaPreprocessor(data_dir=data_dir)

    print("="*70)
    print("PPG-DaLiA Dataset Batch Processing")
    print("="*70)
    print(f"Window size: {window_seconds}s")
    print(f"Stride: {stride_seconds}s")
    print(f"Subjects: {list(subjects)}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    successful = 0
    failed = 0
    failed_subjects = []

    for subject_id in subjects:
        try:
            # Process subject
            processed = preprocessor.process_subject(
                subject_id=subject_id,
                window_seconds=window_seconds,
                stride_seconds=stride_seconds
            )

            # Save
            preprocessor.save_processed_data(processed, output_dir=output_dir)

            successful += 1

        except FileNotFoundError:
            print(f"  ✗ Subject {subject_id}: File not found (skipping)")
            failed += 1
            failed_subjects.append(subject_id)

        except Exception as e:
            print(f"  ✗ Subject {subject_id}: Error - {e}")
            failed += 1
            failed_subjects.append(subject_id)

    print("\n" + "="*70)
    print("Processing Complete")
    print("="*70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_subjects:
        print(f"Failed subjects: {failed_subjects}")
    print("="*70)

    # Summary statistics
    if successful > 0:
        print("\nProcessed data is ready for embedding generation!")
        print(f"Location: {output_dir}/")
        print("\nNext step: Generate embeddings using Wood Wide API")

    return successful, failed


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process PPG-DaLiA dataset subjects"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw data (default: data/raw)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save processed data (default: data/processed)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=30.0,
        help="Window size in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=5.0,
        help="Stride between windows in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        help="Specific subject IDs to process (default: all 1-15)"
    )

    args = parser.parse_args()

    # Process
    successful, failed = process_all_subjects(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_seconds=args.window,
        stride_seconds=args.stride,
        subjects=args.subjects
    )

    # Exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
