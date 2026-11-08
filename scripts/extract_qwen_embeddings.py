"""
Extract text embeddings using Qwen3-Embedding-0.6B for MicroLens item titles.

Corrections vs v1:
- Uses last-token pooling (not first-token) as recommended by Qwen3 docs
- Uses left padding for correct last-token extraction
- Applies L2 normalisation

NOTE: This script was written after the dissertation submission.
The v1 embeddings used in the reported experiments used first-token pooling,
which is incorrect for decoder-only models like Qwen3. Results with this
corrected script have NOT been re-run and may differ from the reported numbers.

Usage:
    python scripts/extract_qwen_embeddings.py \
        --input data/microlens/MicroLens-100k_title_en.csv \
        --output data/microlens/qwen_embeddings_v2/merged_predictions_tensor.pt
"""

import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to titles CSV")
    parser.add_argument("--output", required=True, help="Output path for embeddings .pt")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def last_token_pool(last_hidden_state, attention_mask):
    """Last-token pooling for decoder-only models (correct for Qwen3)."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device),
        sequence_lengths
    ]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    df = pd.read_csv(args.input)
    titles = df.iloc[:, 1].tolist()
    print(f"Loaded {len(titles)} titles")

    all_embeddings = []
    for i in range(0, len(titles), args.batch_size):
        batch = titles[i:i + args.batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

        if i % 1000 == 0:
            print(f"Processed {i}/{len(titles)}")

    result = torch.cat(all_embeddings, dim=0)
    print(f"Final embeddings shape: {result.shape}")
    torch.save(result, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
