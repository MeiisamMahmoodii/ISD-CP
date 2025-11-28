import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import CausalDataset, collate_fn
from src.model.transformer import CausalTransformer
from src.train.trainer import Trainer
from src.utils.monitor import log_gpu_usage
import os
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for training the ISD-CP model.
    
    Steps:
    1. Parse command line arguments.
    2. Initialize the Dataset and DataLoaders.
    3. Initialize the CausalTransformer model.
    4. Setup the Optimizer (AdamW).
    5. Initialize the Trainer.
    6. Run the training loop for specified epochs.
    7. Save model checkpoints.
    """
    parser = argparse.ArgumentParser(description="Train the ISD-CP Causal Transformer")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Directory containing .pt data files")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size (samples per update)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_vars", type=int, default=1000, help="Maximum number of variables (for model embedding)")
    parser.add_argument("--min_vars", type=int, default=10, help="Minimum number of variables per SCM")
    parser.add_argument("--num_scms", type=int, default=100, help="Number of SCMs per epoch")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--micro_batch_size", type=int, default=20, help="Micro batch size to avoid OOM")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable auto-launch of TensorBoard")
    parser.add_argument("--lambda_aux", type=float, default=0.1, help="Weight for auxiliary attention loss")
    parser.add_argument("--lambda_sparse", type=float, default=0.01, help="Weight for L1 sparsity penalty on attention")
    parser.add_argument("--edge_threshold", type=float, default=0.1, help="Threshold for converting attention to edges")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse same SCMs for N epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add file handler to logger
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # Add to root logger so Trainer can use it too if it uses logging
    logging.getLogger().addHandler(file_handler)

    # Launch TensorBoard
    if not args.no_tensorboard:
        import subprocess
        import time
        logger.info("Launching TensorBoard...")
        tb_log_dir = os.path.join(args.output_dir, "logs")
        
        # Try to find tensorboard in the same bin dir as python
        tb_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        if not os.path.exists(tb_executable):
            tb_executable = "tensorboard" # Fallback to PATH
            
        # Run in background
        try:
            subprocess.Popen([tb_executable, "--logdir", tb_log_dir, "--port", "6006"])
            time.sleep(2) # Give it a sec
        except FileNotFoundError:
            logger.warning("TensorBoard executable not found. Skipping auto-launch.")
    
    # 1. Data
    # Use OnlineCausalDataset for infinite/large-scale data without storage
    
    from src.data.dataset import OnlineCausalDataset
    
    # Train set: Generates new SCMs every time
    dataset = OnlineCausalDataset(
        num_samples=args.num_scms, # 100 SCMs per epoch
        min_vars=args.min_vars,
        max_vars=args.max_vars, # Actual SCM size will be random in [min, max]
        n_int_samples=args.batch_size,
        seed=42
    )
    
    # Validation set: Should ideally be fixed or use a different seed range
    # We use a smaller fixed set for validation (generated on fly but deterministic)
    val_dataset = OnlineCausalDataset(
        num_samples=20, 
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        seed=12345 # Different seed
    )
    
    # Initialize DataLoaders
    # num_workers is CRITICAL here. The CPU generates data while GPU trains.
    # set num_workers > 0 (e.g., 4 or 8) to parallelize generation.
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # 2. Model
    # Initialize the Causal Transformer
    # num_vars must be the MAXIMUM possible ID + 1 (or just max_vars)
    model = CausalTransformer(
        num_vars=args.max_vars + 1, # +1 for safety or if 1-indexed (here 0-indexed is fine but safe)
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=0.1
    )
    
    # Load checkpoint if requested
    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_checkpoint}...")
        state_dict = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Checkpoint loaded successfully.")
        
    # 3. Optimizer
    # AdamW is standard for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    # Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 4. Trainer
    # Initialize the Trainer class which manages the loop
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scheduler=scheduler,
        log_dir=os.path.join(args.output_dir, "logs"),
        accumulation_steps=args.accumulation_steps,
        micro_batch_size=args.micro_batch_size,
        lambda_aux=args.lambda_aux,
        lambda_sparse=args.lambda_sparse,
        edge_threshold=args.edge_threshold,
        grad_clip=args.grad_clip
    )
    
    # 5. Loop
    logger.info(f"Starting training on {trainer.device}...")
    log_gpu_usage()
    
    for epoch in range(args.epochs):
        # Update dataset epoch to generate new SCMs
        # If reuse_factor > 1, we stay on the same "dataset epoch" for multiple training epochs
        dataset.set_epoch(epoch // args.reuse_factor)
        
        loss = trainer.train_epoch(epoch)
        val_loss, val_shd, val_f1 = trainer.validate(epoch)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - SHD: {val_shd:.2f} - F1: {val_f1:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt"))
        logger.info(f"Saved checkpoint: model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
