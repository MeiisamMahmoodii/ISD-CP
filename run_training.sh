#!/bin/bash

# ISD-CP Training Script
# This script runs the training with ALL available flags explicitly set.
# You can modify the values here directly.

# ==========================================
# 1. General Configuration
# ==========================================
OUTPUT_DIR="checkpoints_prod"  # Where to save logs and models
#OUTPUT_DIR="checkpoints_5_huber"
#OUTPUT_DIR="checkpoints_5_causal_focus"
NUM_WORKERS=16                    # Number of CPU workers for data generation (Set to 0 for debugging)
RESUME_CHECKPOINT=""             # Path to .pt file to resume from (leave empty to start fresh)
NO_TENSORBOARD=""                # Set to "--no_tensorboard" to disable auto-launch

# ==========================================
# ==========================================
# 2. Data Generation (Infinite Online Data)
# ==========================================
NUM_SCMS=1000       # Increased for throughput
MIN_VARS=10         # Minimum nodes per graph
MAX_VARS=100        # Maximum nodes per graph
REUSE_FACTOR=1      # How many epochs to reuse the same SCMs (1 = new every epoch)

# ==========================================
# 3. Model Architecture
# ==========================================
D_MODEL=256         # Transformer embedding dimension
NUM_LAYERS=8        # Number of Transformer layers

# ==========================================
# 4. Training Hyperparameters
# ==========================================
EPOCHS=1000              # Total training epochs
BATCH_SIZE=4000         # Increased to fill VRAM (was 100)
MICRO_BATCH_SIZE=4000   # No micro-batching needed for small graphs
ACCUMULATION_STEPS=1    # Gradient accumulation steps
LR=1e-4                 # Learning Rate
GRAD_CLIP=1.0           # Gradient Clipping Norm

# ==========================================
# 5. Loss Function & Regularization
# ==========================================
# Options: huber, causal_focus, three_tier
LOSS_FUNCTION="three_tier"
#LOSS_FUNCTION="huber"
#LOSS_FUNCTION="causal_focus" 

# Auxiliary Losses (Usually 0.0 for pure implicit learning)
LAMBDA_AUX=0.0      # Weight for attention supervision
LAMBDA_SPARSE=0.0   # Weight for sparsity penalty
EDGE_THRESHOLD=0.1  # Threshold for metrics

# ==========================================
# Execution Command
# ==========================================
echo "Starting Training..."
echo "Output Directory: $OUTPUT_DIR"
echo "Loss Function: $LOSS_FUNCTION"

python -m src.train.train \
    --output_dir "$OUTPUT_DIR" \
    --num_workers $NUM_WORKERS \
    $NO_TENSORBOARD \
    ${RESUME_CHECKPOINT:+--resume_checkpoint "$RESUME_CHECKPOINT"} \
    --num_scms $NUM_SCMS \
    --min_vars $MIN_VARS \
    --max_vars $MAX_VARS \
    --reuse_factor $REUSE_FACTOR \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --lr $LR \
    --grad_clip $GRAD_CLIP \
    --loss_function "$LOSS_FUNCTION" \
    --lambda_aux $LAMBDA_AUX \
    --lambda_sparse $LAMBDA_SPARSE \
    --edge_threshold $EDGE_THRESHOLD
