import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import CausalDataset, collate_fn
from src.model.transformer import CausalTransformer
from src.train.trainer import Trainer
from src.utils.monitor import log_gpu_usage
import os

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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (not used directly if files are chunks)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_vars", type=int, default=128, help="Number of variables in the SCMs")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Data
    # Load dataset from the specified directory
    dataset = CausalDataset(args.data_dir)
    
    # Initialize DataLoaders
    # We use batch_size=1 because our dataset returns "chunks" (lists of batches)
    # The Trainer handles iterating through these chunks.
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model
    # Initialize the Causal Transformer
    model = CausalTransformer(
        num_vars=args.num_vars,
        d_model=args.d_model,
        num_layers=args.num_layers
    )
    
    # 3. Optimizer
    # AdamW is standard for Transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Trainer
    # Initialize the Trainer class which manages the loop
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 5. Loop
    print(f"Starting training on {trainer.device}...")
    log_gpu_usage()
    
    for epoch in range(args.epochs):
        loss = trainer.train_epoch()
        val_loss = trainer.validate()
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()
