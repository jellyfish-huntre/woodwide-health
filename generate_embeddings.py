"""
Generate embeddings for preprocessed data using Wood Wide API.

This script loads preprocessed window data and generates embeddings
that capture multivariate relationships between heart rate and activity.
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys

from src.embeddings.api_client import APIClient, MockAPIClient


def load_processed_data(subject_id: int, data_dir: str = "data/processed"):
    """Load preprocessed subject data."""
    file_path = Path(data_dir) / f"subject_{subject_id:02d}_processed.pkl"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_embeddings(embeddings: np.ndarray, subject_id: int, output_dir: str = "data/embeddings"):
    """Save generated embeddings."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"subject_{subject_id:02d}_embeddings.npy"

    np.save(output_file, embeddings)
    print(f"✓ Saved embeddings to {output_file}")

    return output_file


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings using Wood Wide API"
    )
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID to process (e.g., 1)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output-dir",
        default="data/embeddings",
        help="Directory to save embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for API requests (default: 32)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: API's default)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock API client (for testing without real API)"
    )

    args = parser.parse_args()

    print("="*70)
    print("Wood Wide Embedding Generation")
    print("="*70)

    try:
        # Load preprocessed data
        print(f"\nLoading subject {args.subject_id}...")
        data = load_processed_data(args.subject_id, args.data_dir)

        windows = data['windows']
        timestamps = data['timestamps']
        labels = data['labels']
        metadata = data['metadata']

        print(f"✓ Loaded {len(windows)} windows")
        print(f"  Window shape: {windows.shape}")
        print(f"  Duration: {metadata['window_seconds']}s per window")

        # Initialize API client
        print("\nInitializing API client...")
        if args.mock:
            print("⚠ Using MOCK API client (no real API calls)")
            client = MockAPIClient(embedding_dim=args.embedding_dim or 128)
        else:
            print("Using real Wood Wide API")
            client = APIClient()

        # Check API health
        print("\nChecking API status...")
        health = client.check_health()
        print(f"✓ API Status: {health.get('status', 'unknown')}")

        # Generate embeddings
        print(f"\nGenerating embeddings...")
        print(f"  Batch size: {args.batch_size}")
        if args.embedding_dim:
            print(f"  Embedding dimension: {args.embedding_dim}")

        embeddings = client.generate_embeddings(
            windows,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim
        )

        print(f"✓ Generated embeddings: {embeddings.shape}")

        # Save embeddings
        print("\nSaving embeddings...")
        output_file = save_embeddings(embeddings, args.subject_id, args.output_dir)

        # Also save metadata
        metadata_file = Path(args.output_dir) / f"subject_{args.subject_id:02d}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'subject_id': args.subject_id,
                'embeddings_shape': embeddings.shape,
                'timestamps': timestamps,
                'labels': labels,
                'window_metadata': metadata
            }, f)
        print(f"✓ Saved metadata to {metadata_file}")

        # Summary
        print("\n" + "="*70)
        print("Embedding Generation Complete")
        print("="*70)
        print(f"Subject: {args.subject_id}")
        print(f"Windows processed: {len(embeddings)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Output location: {output_file}")
        print("\nNext step: Analyze embeddings for signal decoupling")

        client.close()
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run preprocessing first:")
        print(f"  python3 -m src.ingestion.preprocess")
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
