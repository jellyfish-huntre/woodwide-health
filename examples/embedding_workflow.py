"""
Complete workflow example: From raw windows to embeddings.

This script demonstrates the full pipeline for generating embeddings
using the Wood Wide API.
"""

import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.generate import (
    send_windows_to_woodwide,
    process_subject_windows,
    batch_process_subjects,
    load_embeddings
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """
    Example 1: Basic usage with mock API.

    Demonstrates the simplest way to generate embeddings.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)

    # Load preprocessed windows (or create synthetic)
    windows = np.random.randn(50, 960, 5).astype(np.float32)

    # Send to Wood Wide API
    embeddings, metadata = send_windows_to_woodwide(
        windows,
        batch_size=16,
        embedding_dim=128,
        use_mock=True  # Use mock for testing
    )

    print(f"\nInput shape: {windows.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Processing time: {metadata['processing_time_seconds']:.2f}s")
    print(f"Throughput: {metadata['windows_per_second']:.1f} windows/sec")


def example_2_custom_configuration():
    """
    Example 2: Custom configuration.

    Demonstrates advanced configuration options.
    """
    print("\n" + "="*70)
    print("Example 2: Custom Configuration")
    print("="*70)

    windows = np.random.randn(100, 960, 5).astype(np.float32)

    # Custom settings
    embeddings, metadata = send_windows_to_woodwide(
        windows,
        batch_size=64,           # Larger batches
        embedding_dim=256,        # Higher dimension
        validate_input=True,      # Enable validation
        use_mock=True
    )

    print(f"\nConfiguration:")
    print(f"  Batch size: {metadata['batch_size']}")
    print(f"  Embedding dim: {metadata['embedding_dim']}")
    print(f"  Windows: {metadata['n_windows']}")
    print(f"\nPerformance:")
    print(f"  Time: {metadata['processing_time_seconds']:.2f}s")
    print(f"  Throughput: {metadata['windows_per_second']:.1f} windows/sec")


def example_3_process_subject():
    """
    Example 3: Process complete subject.

    Demonstrates end-to-end workflow for a preprocessed subject.
    """
    print("\n" + "="*70)
    print("Example 3: Process Complete Subject")
    print("="*70)

    try:
        # Process subject 1 with all steps: load → validate → embed → save
        embeddings, metadata = process_subject_windows(
            subject_id=1,
            data_dir="data/processed",
            output_dir="data/embeddings",
            batch_size=32,
            embedding_dim=128,
            use_mock=True
        )

        print(f"\nSubject {metadata['subject_id']} processed:")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Windows: {metadata['generation_metadata']['n_windows']}")
        print(f"  Processing time: {metadata['generation_metadata']['processing_time_seconds']:.2f}s")
        print(f"\nSaved to:")
        print(f"  data/embeddings/subject_01_embeddings.npy")
        print(f"  data/embeddings/subject_01_metadata.pkl")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nRun preprocessing first:")
        print("  python -m src.ingestion.preprocess")


def example_4_load_embeddings():
    """
    Example 4: Load previously generated embeddings.

    Demonstrates how to load embeddings for analysis.
    """
    print("\n" + "="*70)
    print("Example 4: Load Embeddings")
    print("="*70)

    try:
        # Load embeddings
        embeddings, metadata = load_embeddings(
            subject_id=1,
            data_dir="data/embeddings"
        )

        print(f"\nLoaded embeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Subject: {metadata['subject_id']}")
        print(f"  Timestamps: {len(metadata['timestamps'])} points")
        print(f"  Unique activities: {np.unique(metadata['labels'])}")

        # Example: Compute similarity between first and last window
        first_emb = embeddings[0]
        last_emb = embeddings[-1]
        similarity = np.dot(first_emb, last_emb)  # Cosine similarity (unit vectors)

        print(f"\nExample analysis:")
        print(f"  Similarity between first and last window: {similarity:.3f}")
        print(f"  (1.0 = identical, -1.0 = opposite)")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nGenerate embeddings first:")
        print("  python -m src.embeddings.generate")


def example_5_batch_processing():
    """
    Example 5: Batch process multiple subjects.

    Demonstrates processing multiple subjects in one call.
    """
    print("\n" + "="*70)
    print("Example 5: Batch Processing")
    print("="*70)

    try:
        # Process subjects 1, 2, 3
        results = batch_process_subjects(
            subject_ids=[1, 2, 3],
            batch_size=32,
            use_mock=True
        )

        print(f"\nBatch processing complete:")
        print(f"  Processed: {len(results)} subjects")

        for subject_id, (embeddings, metadata) in results.items():
            print(f"\n  Subject {subject_id}:")
            print(f"    Embeddings: {embeddings.shape}")
            print(f"    Time: {metadata['generation_metadata']['processing_time_seconds']:.2f}s")

    except FileNotFoundError:
        print("\nNote: Some subjects may not have preprocessed data yet.")
        print("Run: python process_all_subjects.py")


def example_6_validation():
    """
    Example 6: Input validation.

    Demonstrates automatic validation of input data.
    """
    print("\n" + "="*70)
    print("Example 6: Input Validation")
    print("="*70)

    # Valid windows
    valid_windows = np.random.randn(10, 960, 5).astype(np.float32)

    print("\n1. Valid windows:")
    try:
        embeddings, _ = send_windows_to_woodwide(
            valid_windows,
            validate_input=True,
            use_mock=True
        )
        print(f"  ✓ Validation passed, generated {len(embeddings)} embeddings")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")

    # Invalid windows (wrong shape)
    print("\n2. Invalid shape:")
    try:
        invalid_windows = np.random.randn(10, 960)  # 2D instead of 3D
        embeddings, _ = send_windows_to_woodwide(
            invalid_windows,
            validate_input=True,
            use_mock=True
        )
        print(f"  ✓ Unexpected success")
    except ValueError as e:
        print(f"  ✗ Validation caught error: must be 3D array")

    # Windows with NaN
    print("\n3. Windows with NaN values:")
    try:
        nan_windows = np.random.randn(10, 960, 5).astype(np.float32)
        nan_windows[0, 0, 0] = np.nan
        embeddings, _ = send_windows_to_woodwide(
            nan_windows,
            validate_input=True,
            use_mock=True
        )
        print(f"  ✓ Unexpected success")
    except ValueError as e:
        print(f"  ✗ Validation caught error: NaN values detected")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Wood Wide API - Embedding Generation Examples")
    print("="*70)

    examples = [
        example_1_basic_usage,
        example_2_custom_configuration,
        example_3_process_subject,
        example_4_load_embeddings,
        example_5_batch_processing,
        example_6_validation
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nExample {i} error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
