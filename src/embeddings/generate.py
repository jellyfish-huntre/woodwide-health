"""
High-level functions for generating embeddings from windowed health data.

This module provides convenient functions for the complete embedding generation
workflow: validation, API communication, and post-processing.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import pickle
import time

from .api_client import APIClient, MockAPIClient, WoodWideAPIError

logger = logging.getLogger(__name__)


def send_windows_to_woodwide(
    windows: np.ndarray,
    api_key: Optional[str] = None,
    batch_size: int = 32,
    embedding_dim: Optional[int] = None,
    validate_input: bool = True,
    use_mock: bool = False,
    progress_callback=None
) -> Tuple[np.ndarray, Dict]:
    """
    Send windowed time-series data to Wood Wide API and retrieve embeddings.

    This is the main function for transforming preprocessed health data into
    multivariate embeddings that capture contextual relationships between
    heart rate and physical activity.

    Args:
        windows: Windowed time-series data
                 Shape: (n_windows, window_length, n_features)
                 Expected features: [PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG]
        api_key: Wood Wide API key (optional if set in environment)
        batch_size: Number of windows to send per API request
                   Larger = faster but more memory
                   Recommended: 32 for most cases, 16 for large datasets
        embedding_dim: Target embedding dimension (None = API default)
        validate_input: Whether to validate input data format and ranges
        use_mock: Use mock client instead of real API (for testing)
        progress_callback: Optional callback(step, message) for progress updates.
                          Passed through to APIClient.generate_embeddings().

    Returns:
        Tuple of (embeddings, metadata):
        - embeddings: Array of shape (n_windows, embedding_dim)
        - metadata: Dictionary with generation statistics and info

    Raises:
        ValueError: If input validation fails
        WoodWideAPIError: If API communication fails

    Example:
        >>> windows = load_preprocessed_data('subject_01')
        >>> embeddings, meta = send_windows_to_woodwide(
        ...     windows,
        ...     batch_size=32,
        ...     embedding_dim=128
        ... )
        >>> print(f"Generated {len(embeddings)} embeddings")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
    """
    start_time = time.time()

    # Step 1: Validate input
    if validate_input:
        logger.info("Validating input data...")
        _validate_windows(windows)

    n_windows, window_length, n_features = windows.shape
    logger.info(f"Sending {n_windows} windows to Wood Wide API")
    logger.info(f"Window shape: ({window_length}, {n_features})")
    logger.info(f"Batch size: {batch_size}")

    # Step 2: Initialize API client
    if use_mock:
        logger.warning("Using MOCK API client - no real API calls will be made")
        client = MockAPIClient(embedding_dim=embedding_dim or 128)
    else:
        logger.info("Initializing Wood Wide API client")
        client = APIClient(api_key=api_key)

    try:
        # Step 3: Check API health
        logger.info("Checking API health...")
        health = client.check_health()
        logger.info(f"API status: {health.get('status', 'unknown')}")

        # Step 4: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = client.generate_embeddings(
            windows,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            progress_callback=progress_callback
        )

        # Step 5: Validate output
        logger.info("Validating embeddings...")
        _validate_embeddings(embeddings, n_windows)

        # Step 6: Compute metadata
        elapsed_time = time.time() - start_time
        metadata = {
            'n_windows': n_windows,
            'window_shape': (window_length, n_features),
            'embedding_dim': embeddings.shape[1],
            'batch_size': batch_size,
            'processing_time_seconds': elapsed_time,
            'windows_per_second': n_windows / elapsed_time,
            'api_status': health.get('status'),
            'mock': use_mock
        }

        logger.info(f"✓ Successfully generated {len(embeddings)} embeddings")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  Processing time: {elapsed_time:.2f}s")
        logger.info(f"  Throughput: {metadata['windows_per_second']:.1f} windows/sec")

        return embeddings, metadata

    except WoodWideAPIError as e:
        logger.error(f"API error during embedding generation: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}")
        raise

    finally:
        client.close()


def _validate_windows(windows: np.ndarray) -> None:
    """
    Validate windowed data format and ranges.

    Args:
        windows: Array to validate

    Raises:
        ValueError: If validation fails
    """
    # Check dimensionality
    if windows.ndim != 3:
        raise ValueError(
            f"Windows must be 3D array (n_windows, window_length, n_features), "
            f"got {windows.ndim}D array with shape {windows.shape}"
        )

    n_windows, window_length, n_features = windows.shape

    # Check minimum windows
    if n_windows == 0:
        raise ValueError("No windows to process (n_windows = 0)")

    # Check expected features
    if n_features != 5:
        logger.warning(
            f"Expected 5 features (PPG, ACC_X, ACC_Y, ACC_Z, ACC_MAG), "
            f"got {n_features}. Proceeding anyway."
        )

    # Check for null values
    if np.isnan(windows).any():
        n_nulls = np.isnan(windows).sum()
        raise ValueError(f"Found {n_nulls} null/NaN values in windows")

    # Check for infinite values
    if np.isinf(windows).any():
        n_infs = np.isinf(windows).sum()
        raise ValueError(f"Found {n_infs} infinite values in windows")

    # Check data ranges (warnings only)
    windows_flat = windows.reshape(-1, n_features)
    for i in range(n_features):
        feature_data = windows_flat[:, i]
        min_val, max_val = feature_data.min(), feature_data.max()

        # Very rough sanity checks
        if abs(min_val) > 1000 or abs(max_val) > 1000:
            logger.warning(
                f"Feature {i} has extreme values: [{min_val:.2f}, {max_val:.2f}]. "
                f"Consider checking preprocessing."
            )

    logger.info(f"✓ Input validation passed")


def _validate_embeddings(embeddings: np.ndarray, expected_n_windows: int) -> None:
    """
    Validate embedding output.

    Args:
        embeddings: Generated embeddings
        expected_n_windows: Expected number of windows

    Raises:
        ValueError: If validation fails
    """
    # Check shape
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D array (n_windows, embedding_dim), "
            f"got {embeddings.ndim}D"
        )

    n_embeddings, embedding_dim = embeddings.shape

    # Check count
    if n_embeddings != expected_n_windows:
        raise ValueError(
            f"Expected {expected_n_windows} embeddings, got {n_embeddings}"
        )

    # Check for null values
    if np.isnan(embeddings).any():
        n_nulls = np.isnan(embeddings).sum()
        raise ValueError(f"Found {n_nulls} null/NaN values in embeddings")

    # Check normalization (embeddings should typically be unit normalized)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3):
        logger.warning(
            f"Embeddings may not be unit-normalized. "
            f"Norm range: [{norms.min():.3f}, {norms.max():.3f}]"
        )

    logger.info(f"✓ Embedding validation passed")


def process_subject_windows(
    subject_id: int,
    data_dir: str = "data/processed",
    output_dir: str = "data/embeddings",
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Load preprocessed subject data, generate embeddings, and save results.

    This is a convenience function that handles the complete workflow:
    load → validate → generate embeddings → save.

    Args:
        subject_id: Subject ID (1-15 for PPG-DaLiA)
        data_dir: Directory containing preprocessed data
        output_dir: Directory to save embeddings
        **kwargs: Additional arguments passed to send_windows_to_woodwide()

    Returns:
        Tuple of (embeddings, metadata)

    Example:
        >>> embeddings, meta = process_subject_windows(
        ...     subject_id=1,
        ...     batch_size=32,
        ...     use_mock=True
        ... )
    """
    # Load preprocessed data
    logger.info(f"Loading subject {subject_id}...")
    input_file = Path(data_dir) / f"subject_{subject_id:02d}_processed.pkl"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {input_file}\n"
            f"Run preprocessing first: python -m src.ingestion.preprocess"
        )

    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    windows = data['windows']
    timestamps = data['timestamps']
    labels = data['labels']
    window_metadata = data['metadata']

    logger.info(f"✓ Loaded {len(windows)} windows")

    # Generate embeddings
    embeddings, gen_metadata = send_windows_to_woodwide(windows, **kwargs)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy array
    embeddings_file = output_path / f"subject_{subject_id:02d}_embeddings.npy"
    np.save(embeddings_file, embeddings)
    logger.info(f"✓ Saved embeddings: {embeddings_file}")

    # Save metadata
    metadata_file = output_path / f"subject_{subject_id:02d}_metadata.pkl"
    full_metadata = {
        'subject_id': subject_id,
        'embeddings_shape': embeddings.shape,
        'timestamps': timestamps,
        'labels': labels,
        'window_metadata': window_metadata,
        'generation_metadata': gen_metadata
    }

    with open(metadata_file, 'wb') as f:
        pickle.dump(full_metadata, f)
    logger.info(f"✓ Saved metadata: {metadata_file}")

    return embeddings, full_metadata


def batch_process_subjects(
    subject_ids: List[int],
    **kwargs
) -> Dict[int, Tuple[np.ndarray, Dict]]:
    """
    Process multiple subjects in batch.

    Args:
        subject_ids: List of subject IDs to process
        **kwargs: Arguments passed to process_subject_windows()

    Returns:
        Dictionary mapping subject_id -> (embeddings, metadata)

    Example:
        >>> results = batch_process_subjects(
        ...     subject_ids=[1, 2, 3],
        ...     batch_size=32,
        ...     use_mock=True
        ... )
        >>> for subject_id, (embeddings, meta) in results.items():
        ...     print(f"Subject {subject_id}: {embeddings.shape}")
    """
    results = {}
    failed = []

    for subject_id in subject_ids:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing Subject {subject_id}")
            logger.info(f"{'='*70}")

            embeddings, metadata = process_subject_windows(subject_id, **kwargs)
            results[subject_id] = (embeddings, metadata)

            logger.info(f"✓ Subject {subject_id} complete")

        except Exception as e:
            logger.error(f"✗ Subject {subject_id} failed: {e}")
            failed.append(subject_id)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Batch Processing Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Successful: {len(results)}/{len(subject_ids)}")
    if failed:
        logger.info(f"Failed: {failed}")

    return results


def load_embeddings(
    subject_id: int,
    data_dir: str = "data/embeddings"
) -> Tuple[np.ndarray, Dict]:
    """
    Load previously generated embeddings.

    Args:
        subject_id: Subject ID
        data_dir: Directory containing embeddings

    Returns:
        Tuple of (embeddings, metadata)

    Example:
        >>> embeddings, meta = load_embeddings(1)
        >>> print(f"Loaded {len(embeddings)} embeddings")
    """
    embeddings_file = Path(data_dir) / f"subject_{subject_id:02d}_embeddings.npy"
    metadata_file = Path(data_dir) / f"subject_{subject_id:02d}_metadata.pkl"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")

    embeddings = np.load(embeddings_file)

    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

    logger.info(f"✓ Loaded embeddings for subject {subject_id}: {embeddings.shape}")

    return embeddings, metadata


if __name__ == "__main__":
    """Example usage."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Generate embeddings for subject 1 using mock API
    try:
        embeddings, metadata = process_subject_windows(
            subject_id=1,
            batch_size=32,
            embedding_dim=128,
            use_mock=True
        )

        print("\n" + "="*70)
        print("Embedding Generation Complete")
        print("="*70)
        print(f"Shape: {embeddings.shape}")
        print(f"Processing time: {metadata['generation_metadata']['processing_time_seconds']:.2f}s")
        print(f"Throughput: {metadata['generation_metadata']['windows_per_second']:.1f} windows/sec")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run preprocessing first:")
        print("  python -m src.ingestion.preprocess")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
