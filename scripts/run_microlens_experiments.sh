#!/bin/bash
# MicroLens-100K Experiment Run Commands
# Run from the GRID repo root with: conda activate grid

# ── Video Modality (VideoMAE embeddings, 768-dim) ──────────────────────────

# Step 1: Train RQ-VAE on VideoMAE embeddings
python -m src.train experiment=rqvae_train_flat \
    data_dir=data/microlens \
    embedding_path=data/microlens/videomae_embeddings/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.max_steps=10000

# Step 2: Run RQ-VAE inference to generate Semantic IDs
python -m src.inference experiment=rqvae_inference_flat \
    data_dir=data/microlens \
    embedding_path=data/microlens/videomae_embeddings/merged_predictions_tensor.pt \
    embedding_dim=768 \
    num_hierarchies=3 \
    codebook_width=256

# Step 3: Transpose Semantic IDs for TIGER
python scripts/transpose_semantic_ids.py \
    --input logs/inference/runs/<date>/pickle/merged_predictions_tensor.pt \
    --output data/microlens/semantic_ids_video.pt

# Step 4: Train TIGER on video Semantic IDs
python -m src.train experiment=tiger_train_flat \
    data_dir=data/microlens \
    semantic_id_path=data/microlens/semantic_ids_video.pt \
    num_hierarchies=4 \
    trainer.devices=1 \
    callbacks.early_stopping.patience=20

# ── Text Modality (Qwen3-Embedding-0.6B, 1024-dim) ────────────────────────

# Step 1: Extract Qwen embeddings (corrected v2 - not yet re-run)
python scripts/extract_qwen_embeddings.py \
    --input data/microlens/MicroLens-100k_title_en.csv \
    --output data/microlens/qwen_embeddings_v2/merged_predictions_tensor.pt

# Step 2: Train RQ-VAE on Qwen embeddings
python -m src.train experiment=rqvae_train_flat \
    data_dir=data/microlens \
    embedding_path=data/microlens/qwen_embeddings_v2/merged_predictions_tensor.pt \
    embedding_dim=1024 \
    num_hierarchies=3 \
    codebook_width=256 \
    trainer.max_steps=10000

# Step 3: Run RQ-VAE inference
python -m src.inference experiment=rqvae_inference_flat \
    data_dir=data/microlens \
    embedding_path=data/microlens/qwen_embeddings_v2/merged_predictions_tensor.pt \
    embedding_dim=1024 \
    num_hierarchies=3 \
    codebook_width=256

# Step 4: Transpose Semantic IDs
python scripts/transpose_semantic_ids.py \
    --input logs/inference/runs/<date>/pickle/merged_predictions_tensor.pt \
    --output data/microlens/semantic_ids_text.pt

# Step 5: Train TIGER on text Semantic IDs
python -m src.train experiment=tiger_train_flat \
    data_dir=data/microlens \
    semantic_id_path=data/microlens/semantic_ids_text.pt \
    num_hierarchies=4 \
    trainer.devices=1 \
    callbacks.early_stopping.patience=20
