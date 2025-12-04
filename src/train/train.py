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
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Fix for tmux/multiprocessing crash
# This resolves "RuntimeError: received 0 items of ancdata" when running in tmux
torch.multiprocessing.set_sharing_strategy('file_system')

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
    parser.add_argument("--lambda_aux", type=float, default=0.0, help="Weight for auxiliary attention loss")
    parser.add_argument("--lambda_sparse", type=float, default=0.0, help="Weight for L1 sparsity penalty on attention")
    parser.add_argument("--edge_threshold", type=float, default=0.1, help="Threshold for converting attention to edges")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse same SCMs for N epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--loss_function", type=str, default="huber", choices=["huber", "causal_focus", "three_tier"], help="Loss function to use")
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
        import socket
        
        def find_free_port(start_port=6000):
            port = start_port
            while port < 65535:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('', port))
                        return port
                except OSError:
                    port += 1
            raise RuntimeError("No free ports found")
        
        logger.info("Launching TensorBoard...")
        tb_log_dir = os.path.join(args.output_dir, "logs")
        
        # Try to find tensorboard in the same bin dir as python
        tb_executable = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        if not os.path.exists(tb_executable):
            tb_executable = "tensorboard" # Fallback to PATH
            
        # Run in background
        try:
            port = find_free_port(6000)
            subprocess.Popen([tb_executable, "--logdir", tb_log_dir, "--port", str(port), "--bind_all"])

            time.sleep(2) # Give it a sec
            logger.info(f"TensorBoard launched at http://localhost:{port}")
        except FileNotFoundError:
            logger.warning("TensorBoard executable not found. Skipping auto-launch.")
        except Exception as e:
            logger.warning(f"Failed to launch TensorBoard: {e}")
    
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
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None)
    
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
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        loss_fn_name=args.loss_function
    )
    
    # 5. Loop
    logger.info(f"Starting training on {trainer.device}...")
    log_gpu_usage()
    
    trainer.fit()



if __name__ == "__main__":
    main()
