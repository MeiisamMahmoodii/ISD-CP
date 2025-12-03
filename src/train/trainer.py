import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from src.utils.metrics import compute_shd, compute_f1, extract_attention_dag, plot_adjacency_heatmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Trainer:
    """
    Handles the training and validation loop for the CausalTransformer.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler = None,
        log_dir: str = "logs",
        accumulation_steps: int = 1,
        micro_batch_size: int = 20, # Default safe size
        lambda_aux: float = 0.1, # Weight for auxiliary attention loss
        lambda_sparse: float = 0.01, # Weight for sparsity penalty
        edge_threshold: float = 0.1, # Threshold for edge detection
        grad_clip: float = 1.0,   # Gradient clipping value
        epochs: int = 100 # Total epochs for annealing
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.micro_batch_size = micro_batch_size
        self.lambda_aux = lambda_aux
        self.lambda_sparse = lambda_sparse
        self.edge_threshold = edge_threshold
        self.grad_clip = grad_clip
        self.epochs = epochs
        
        self.criterion = nn.HuberLoss(delta=1.0)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        
    def train_epoch(self, epoch_idx):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_pred_loss = 0.0
        total_aux_loss = 0.0
        total_sparse_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)
        postfix_dict = {}
        
        for batch in pbar:
            # Handle nested list structure from DataLoader
            file_content = batch
            if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                 file_content = batch[0]
            
            # Iterate over the mini-batches contained in the loaded file/chunk
            # inner_pbar = tqdm(file_content, desc="Processing SCM", leave=False)
            # Using just one bar might be cleaner if inner loop is fast
            for mini_batch in file_content:
                x = mini_batch['x'].to(self.device)
                mask = mini_batch['mask'].to(self.device)
                value = mini_batch['value'].to(self.device)
                target = mini_batch['target'].to(self.device)
                
                # Squeeze the extra batch dimension added by DataLoader
                if x.ndim == 3 and x.shape[0] == 1:
                    x = x.squeeze(0)
                    mask = mask.squeeze(0)
                    value = value.squeeze(0)
                    target = target.squeeze(0)
                
                # Get Ground Truth Adjacency if available
                true_adj = None
                if 'adj' in mini_batch:
                    true_adj = mini_batch['adj'].to(self.device)
                    if true_adj.ndim == 3 and true_adj.shape[0] == 1:
                        true_adj = true_adj.squeeze(0)
                
                # Micro-batching to avoid OOM
                batch_size = x.shape[0]
                num_micro_batches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size
                
                batch_loss = 0.0
                
                self.optimizer.zero_grad()
                
                # Gumbel Annealing Removed


                for i in range(num_micro_batches):
                    start_idx = i * self.micro_batch_size
                    end_idx = min((i + 1) * self.micro_batch_size, batch_size)
                    
                    x_micro = x[start_idx:end_idx]
                    mask_micro = mask[start_idx:end_idx]
                    value_micro = value[start_idx:end_idx]
                    target_micro = target[start_idx:end_idx]
                    
                    # Forward pass
                    pred_delta, adj = self.model(x_micro, mask_micro, value_micro)
                    
                    # Compute Prediction Loss (Delta Reward)
                    target_delta = target_micro - x_micro
                    loss_pred = self.criterion(pred_delta, target_delta)
                    
                    # Structure Loss Removed (TabICL Encoder)
                    loss_micro = loss_pred
                    
                    # Check for NaNs in loss components
                    if torch.isnan(loss_micro) or torch.isinf(loss_micro):
                        print(f"NaN/Inf Loss detected at step {self.global_step}!")
                        print(f"Pred: {loss_pred.item()}, Aux: {loss_aux.item()}, Sparse: {loss_sparse.item()}, DAG: {loss_dag.item()}")
                        # Skip backward to avoid polluting grads
                        continue
                    
                    # Normalize loss for accumulation
                    weight = (end_idx - start_idx) / batch_size
                    loss_scaled = loss_micro * weight / self.accumulation_steps
                    
                    loss_scaled.backward()
                    
                    batch_loss += loss_micro.item() * weight
                    total_pred_loss += loss_pred.item() * weight
                
                # Optimization step
                if (self.global_step + 1) % self.accumulation_steps == 0:
                    # Gradient Clipping
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    # Check for NaNs in gradients
                    has_nan = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan = True
                                # print(f"WARNING: NaN gradient in {name}")
                                break
                    
                    if not has_nan:
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                    else:
                        print(f"Skipping step {self.global_step} due to NaN gradients.")
                        
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    grad_norm = 0.0
                    if self.grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    # Log to TensorBoard
                    if self.global_step % 50 == 0:
                        self.writer.add_scalar("Loss/Train", loss_micro.item(), self.global_step)
                        self.writer.add_scalar("Loss/Train_Pred", loss_pred.item(), self.global_step)
                        # Aux/Sparse Loss Removed
                        self.writer.add_scalar("System/GradNorm", grad_norm, self.global_step)
                        self.writer.add_scalar("System/LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                        
                        # Compute Metrics for Progress Bar (on first sample of batch)
                        # DAG Metrics Removed
                        postfix_dict.update({
                            'grad': f"{grad_norm:.1f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
                        })

            total_loss += batch_loss
            self.global_step += 1
            
            # Always update loss in postfix_dict
            postfix_dict['loss'] = f"{batch_loss:.2f}"
            pbar.set_postfix(postfix_dict)
            
            # Log batch loss occasionally
            if self.global_step % 50 == 0:
                # Log progress text (Terminal Output in TensorBoard)
                # Log progress text (Terminal Output in TensorBoard)
                # Use Markdown code block for terminal look
                self.writer.add_text("Logs/Terminal", f"```\nEpoch {epoch_idx}: Step {self.global_step}, Loss: {batch_loss:.4f}\n```", self.global_step)
            
            # Log Distributions occasionally
            if self.global_step % 200 == 0:
                self.writer.add_histogram("Dist/Delta_Pred", pred_delta, self.global_step)
                self.writer.add_histogram("Dist/Delta_True", target_delta, self.global_step)
        
        if self.scheduler:
            self.scheduler.step()
            
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/TrainEpoch", avg_loss, epoch_idx)
        return avg_loss

    def log_terminal_message(self, message, step):
        """Logs a text message to TensorBoard."""
        self.writer.add_text("Logs/Terminal", f"```\n{message}\n```", step)

    def validate(self, epoch_idx):
        """
        Runs validation on the validation set.
        Computes Loss only (No DAG metrics).
        """
        self.model.eval()
        total_loss = 0.0
        
        num_graphs = 0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                file_content = batch
                if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                     file_content = batch[0]
                
                first_batch = True
                
                for mini_batch in file_content:
                    x = mini_batch['x'].to(self.device)
                    mask = mini_batch['mask'].to(self.device)
                    value = mini_batch['value'].to(self.device)
                    target = mini_batch['target'].to(self.device)
                    
                    if x.ndim == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                        mask = mask.squeeze(0)
                        value = value.squeeze(0)
                        target = target.squeeze(0)
                    
                    pred_delta, adj = self.model(x, mask, value)
                    
                    # Compute Delta Target
                    target_delta = target - x
                    loss = self.criterion(pred_delta, target_delta)
                    total_loss += loss.item()
                    total_mae += torch.abs(pred_delta - target_delta).mean().item()
                    
                    # Log Visualizations (only for the first graph of the epoch)
                    if first_batch:
                        num_graphs += 1
                        if num_graphs == 1:
                            # 1. Scatter Plot (Pred vs True Delta)
                            fig2, ax = plt.subplots(figsize=(6, 6))
                            t_flat = target_delta.cpu().numpy().flatten()
                            p_flat = pred_delta.cpu().numpy().flatten()
                            if len(t_flat) > 1000:
                                idx = np.random.choice(len(t_flat), 1000, replace=False)
                                t_flat = t_flat[idx]
                                p_flat = p_flat[idx]
                                
                            ax.scatter(t_flat, p_flat, alpha=0.3)
                            ax.plot([t_flat.min(), t_flat.max()], [t_flat.min(), t_flat.max()], 'r--')
                            ax.set_xlabel("True Delta")
                            ax.set_ylabel("Predicted Delta")
                            ax.set_title(f"Prediction Correlation (Epoch {epoch_idx})")
                            ax.grid(True)
                            self.writer.add_figure("Analysis/Pred_vs_True", fig2, epoch_idx)
                            plt.close(fig2)
                            
                            # 2. Histograms
                            if not (torch.isnan(pred_delta).any() or torch.isinf(pred_delta).any()):
                                self.writer.add_histogram("Dist/Val_Delta_Pred", pred_delta, epoch_idx)
                                self.writer.add_histogram("Dist/Val_Delta_True", target_delta, epoch_idx)
                            
                        first_batch = False
                    
        avg_loss = total_loss / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        
        self.writer.add_scalar("Loss/Val", avg_loss, epoch_idx)
        self.writer.add_scalar("Metrics/Val_MAE", avg_mae, epoch_idx)
        
        # Return 0 for SHD/F1 compatibility
        return avg_loss, 0.0, 0.0
