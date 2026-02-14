"""
Generate embeddings for preprocessed data using Wood Wide API.

This script loads preprocessed window data and generates embeddings
that capture multivariate relationships between heart rate and activity.
"""

import argparse
import logging
import pickle
import time
import numpy as np
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.embeddings.api_client import APIClient, MockAPIClient

console = Console()


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
    console.print(f"  [green]>[/green] Saved embeddings to {output_file}")

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
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name for the uploaded dataset (auto-generated if not set)"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name for the trained model (auto-generated if not set)"
    )

    args = parser.parse_args()

    # Suppress duplicate logger output — we show progress via rich
    logging.getLogger("src.embeddings.api_client").setLevel(logging.WARNING)
    logging.getLogger("src.embeddings.generate").setLevel(logging.WARNING)

    console.print()
    console.print(Panel.fit(
        "[bold]Wood Wide Embedding Generation[/bold]",
        border_style="blue"
    ))
    console.print()

    try:
        console.print("  Loading data...")
        data = load_processed_data(args.subject_id, args.data_dir)

        windows = data['windows']
        timestamps = data['timestamps']
        labels = data['labels']
        metadata = data['metadata']

        n_windows, window_length, n_features = windows.shape
        console.print(
            f"  [green]>[/green] Loaded {n_windows} windows "
            f"({window_length} x {n_features}, "
            f"{metadata['window_seconds']}s per window)"
        )

        if args.mock:
            console.print("  [yellow]![/yellow] Using MOCK API client")
            client = MockAPIClient(embedding_dim=args.embedding_dim or 128)
        else:
            console.print("  Connecting to Wood Wide API...")
            client = APIClient()
            console.print(f"  [green]>[/green] Connected to {client.base_url}")

        health = client.check_health()
        console.print(f"  [green]>[/green] API status: {health.get('status', 'unknown')}")
        console.print()

        # Generate embeddings with live progress
        gen_start = time.time()

        with console.status("", spinner="dots") as status:
            def on_progress(step, message):
                elapsed = time.time() - gen_start
                if step == "poll_status":
                    status.update(f"  [cyan]{message}[/cyan]  [dim]({elapsed:.0f}s)[/dim]")
                elif step == "poll_done":
                    status.update(f"  [green]{message}[/green]")
                elif step == "done":
                    pass  # handled below
                else:
                    status.update(f"  [cyan]{message}[/cyan]  [dim]({elapsed:.0f}s)[/dim]")

            embeddings = client.generate_embeddings(
                windows,
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                progress_callback=on_progress
            )

        gen_elapsed = time.time() - gen_start
        console.print(
            f"  [bold green]>[/bold green] Generated {len(embeddings)} embeddings "
            f"({embeddings.shape[1]}-dim) in {gen_elapsed:.1f}s"
        )
        console.print()

        console.print("  Saving results...")
        output_file = save_embeddings(embeddings, args.subject_id, args.output_dir)

        metadata_file = Path(args.output_dir) / f"subject_{args.subject_id:02d}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'subject_id': args.subject_id,
                'embeddings_shape': embeddings.shape,
                'timestamps': timestamps,
                'labels': labels,
                'window_metadata': metadata
            }, f)
        console.print(f"  [green]>[/green] Saved metadata to {metadata_file}")
        console.print()

        # Summary
        throughput = len(embeddings) / gen_elapsed if gen_elapsed > 0 else 0
        summary = Text()
        summary.append(f"  Subject: {args.subject_id}\n")
        summary.append(f"  Windows: {len(embeddings)}  ")
        summary.append(f"  Embedding dim: {embeddings.shape[1]}  ")
        summary.append(f"  Time: {gen_elapsed:.1f}s\n")
        summary.append(f"  Throughput: {throughput:.1f} windows/sec\n")
        summary.append(f"  Output: {output_file}")

        console.print(Panel(summary, title="[bold green]Complete[/bold green]", border_style="green"))
        console.print()

        client.close()
        return 0

    except FileNotFoundError as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        console.print("\n  Please run preprocessing first:")
        console.print("    python3 -m src.ingestion.preprocess")
        return 1

    except Exception as e:
        console.print(f"\n  [bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
