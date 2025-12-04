#!/bin/bash

# Comparison Experiment: Loss Functions
# Runs 3 short training jobs to compare convergence.

# Common Settings (Fast Run)
EPOCHS=5
NUM_SCMS=20
MIN_VARS=5
MAX_VARS=5
BATCH_SIZE=100
NUM_WORKERS=8

echo "========================================"
echo "1. Running Standard Huber Loss..."
echo "========================================"
python -m src.train.train \
    --output_dir checkpoints_compare_huber \
    --loss_function huber \
    --epochs $EPOCHS \
    --num_scms $NUM_SCMS \
    --min_vars $MIN_VARS \
    --max_vars $MAX_VARS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --no_tensorboard

echo "========================================"
echo "2. Running Causal Focus Loss..."
echo "========================================"
python -m src.train.train \
    --output_dir checkpoints_compare_focus \
    --loss_function causal_focus \
    --epochs $EPOCHS \
    --num_scms $NUM_SCMS \
    --min_vars $MIN_VARS \
    --max_vars $MAX_VARS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --no_tensorboard

echo "========================================"
echo "3. Running Three-Tier Loss..."
echo "========================================"
python -m src.train.train \
    --output_dir checkpoints_compare_tier \
    --loss_function three_tier \
    --epochs $EPOCHS \
    --num_scms $NUM_SCMS \
    --min_vars $MIN_VARS \
    --max_vars $MAX_VARS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --no_tensorboard

echo "========================================"
echo "Comparison Complete."
echo "========================================"
