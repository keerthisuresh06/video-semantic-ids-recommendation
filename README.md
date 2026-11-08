# Video-Modality Semantic IDs for Generative Recommendation

MSc Dissertation — Keerthi Suresh — University of Glasgow — 2026

## Overview

This project investigates whether video-modality embeddings produce better
Semantic IDs than text embeddings for generative recommendation on short-video
datasets. Experiments were conducted on MicroLens-100K using the GRID framework.

**Pipeline:**
1. Extract modality embeddings (VideoMAE for video, Qwen3 for text)
2. Train RQ-VAE to generate hierarchical Semantic IDs
3. Train TIGER transformer for next-item prediction
4. Evaluate using Recall@5, Recall@10, NDCG@5, NDCG@10

## Results

| Modality | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|----------|----------|-----------|--------|---------|
| Text (Qwen3-Embedding-0.6B, v1) | 0.0265 | 0.0394 | 0.0176 | 0.0217 |
| Video (VideoMAE) | 0.0309 | 0.0468 | 0.0202 | 0.0253 |

Video modality achieved a relative improvement of ~19% on Recall@10
(0.0074 absolute) over the v1 text baseline. See limitations below
before interpreting this result.

## Repository Structure

The GRID framework code is not duplicated here.
Clone it from: https://github.com/snap-research/GRID

## Setup

```bash
git clone https://github.com/snap-research/GRID
conda activate grid
```

Then place these scripts inside the GRID repo root and follow
run_microlens_experiments.sh.

## Limitations and Known Issues

1. **Text embedding bug (v1):** The Qwen3 embeddings used in the reported
   experiments used first-token pooling. For decoder-only models like Qwen3,
   last-token pooling with left padding is correct. The corrected script
   is `scripts/extract_qwen_embeddings.py` but has not been re-run.
   The ~19% improvement over text should therefore not be taken as a
   reliable comparison — it may partially reflect the pooling bug.

2. **Single seed:** All experiments were run with a single random seed.
   No variance estimates are available. Results may not be reproducible
   exactly.

3. **Short RQ-VAE training:** RQ-VAE was trained for 10,000 steps.
   The original GRID paper used longer schedules. Semantic ID quality
   may improve with more training.

4. **Small Qwen model:** Qwen3-Embedding-0.6B was used due to GPU memory
   constraints. Larger models may produce better text embeddings.

5. **Pre-extracted video features:** VideoMAE features were used as-is
   from the MicroLens dataset release. No fine-tuning was performed on
   the recommendation task.

6. **masking_token and padding_token:** Both are set to -1 in the TIGER
   config, inherited from the base GRID codebase. Masking and padding
   positions are not explicitly distinguished.

## Dataset

Download MicroLens-100K from: https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/
