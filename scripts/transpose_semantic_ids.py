"""
Transpose and deduplicate Semantic IDs output from RQ-VAE inference.

RQ-VAE outputs shape [num_items, num_hierarchies].
TIGER expects shape [num_hierarchies, num_items] with a zero-padded row
for 1-based item indexing used in MicroLens.

Usage:
    python scripts/transpose_semantic_ids.py \
        --input logs/inference/runs/<date>/pickle/merged_predictions_tensor.pt \
        --output data/microlens/semantic_ids_video.pt
"""

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw RQ-VAE output .pt file")
    parser.add_argument("--output", required=True, help="Path to save transposed semantic IDs")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load raw RQ-VAE output: shape [num_items, num_hierarchies]
    ids = torch.load(args.input)
    print(f"Loaded shape: {ids.shape}")

    # Transpose to [num_hierarchies, num_items]
    ids_t = ids.t()
    print(f"Transposed shape: {ids_t.shape}")

    # Add zero-padding row at index 0 for 1-based MicroLens item IDs
    num_hierarchies, num_items = ids_t.shape
    padding = torch.zeros(num_hierarchies, 1, dtype=ids_t.dtype)
    ids_padded = torch.cat([padding, ids_t], dim=1)
    print(f"Padded shape (with zero row): {ids_padded.shape}")

    torch.save(ids_padded, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
