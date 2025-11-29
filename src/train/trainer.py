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
        grad_clip: float = 1.0   # Gradient clipping value
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
        
        self.criterion = nn.MSELoss()
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
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Handle nested list structure from DataLoader
            file_content = batch
            if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                 file_content = batch[0]
            
            # Iterate over the mini-batches contained in the loaded file/chunk
            inner_pbar = tqdm(file_content, desc="Processing SCM", leave=False)
            for mini_batch in inner_pbar:
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
                
                for i in range(num_micro_batches):
                    start_idx = i * self.micro_batch_size
                    end_idx = min((i + 1) * self.micro_batch_size, batch_size)
                    
                    x_micro = x[start_idx:end_idx]
                    mask_micro = mask[start_idx:end_idx]
                    value_micro = value[start_idx:end_idx]
                    target_micro = target[start_idx:end_idx]
                    
                    # Forward pass
                    pred_delta, attn_micro = self.model(x_micro, mask_micro, value_micro)
                    
                    # Compute Prediction Loss (Delta Reward)
                    # Target is the CHANGE from baseline
                    target_delta = target_micro - x_micro
                    loss_pred = self.criterion(pred_delta, target_delta)
                    
                    # Compute Auxiliary Loss (Supervised Attention)
                    loss_aux = torch.tensor(0.0, device=self.device)
                    if true_adj is not None:
                        # true_adj is (num_vars, num_vars) usually, or (batch, num_vars, num_vars) if repeated?
                        # In OnlineCausalDataset, adj is (num_vars, num_vars) repeated for each sample in batch?
                        # No, 'adj' is added to batch_data.
                        # If collate_fn stacks them, it becomes (batch, num_vars, num_vars).
                        # Let's check dimensions.
                        
                        if true_adj.ndim == 2:
                            # Expand to match micro batch size
                            target_adj = true_adj.unsqueeze(0).expand(pred_delta.shape[0], -1, -1)
                        else:
                            # Assuming it's (batch, N, N)
                            target_adj = true_adj[start_idx:end_idx]
                            
                        # attn_micro is (batch, N, N)
                        # We want attn to match adjacency (transposed? A_ij=1 if i->j. Attn_ji means j attends to i)
                        # If j attends to i, then i causes j.
                        # So Attn_ji ~= Adj_ij.
                        # Attn matrix: rows are queries (targets), cols are keys (sources).
                        # Attn[j, i] = weight of i on j.
                        # Adj[i, j] = 1 if i -> j.
                        # So we want Attn[j, i] to be high if Adj[i, j] is 1.
                        # So Target Attn = Adj.T
                        
                        target_attn = target_adj.transpose(1, 2)
                        loss_aux = F.mse_loss(attn_micro, target_attn)
                    
                        loss_aux = F.mse_loss(attn_micro, target_attn)
                    
                    # Compute Sparsity Penalty (L1 on Attention)
                    # We want the attention matrix to be sparse (few edges)
                    loss_sparse = torch.mean(torch.abs(attn_micro))

                    # Total Loss
                    loss_micro = loss_pred + self.lambda_aux * loss_aux + self.lambda_sparse * loss_sparse
                    
                    # Normalize loss for accumulation
                    weight = (end_idx - start_idx) / batch_size
                    loss_scaled = loss_micro * weight / self.accumulation_steps
                    
                    loss_scaled.backward()
                    
                    batch_loss += loss_micro.item() * weight
                    total_pred_loss += loss_pred.item() * weight
                    total_aux_loss += loss_aux.item() * weight
                    total_sparse_loss += loss_sparse.item() * weight
                
                # Optimization step
                if (self.global_step + 1) % self.accumulation_steps == 0:
                    # Gradient Clipping
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Step scheduler if it's per-step (e.g. OneCycleLR)
                    # Assuming scheduler is per-epoch for now, or handled outside.
                    # But if it's CosineAnnealingWarmRestarts, it might be per batch?
                    # Let's stick to per-epoch for now unless specified.
                
                total_loss += batch_loss
                self.global_step += 1
                
                # Log batch loss occasionally
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("Loss/TrainBatch", batch_loss, self.global_step)
                    self.writer.add_scalar("Loss/TrainBatch", batch_loss, self.global_step)
                    self.writer.add_scalar("Loss/Aux", total_aux_loss / (self.global_step % len(file_content) + 1), self.global_step) # Approx
                    self.writer.add_scalar("Loss/Sparse", total_sparse_loss / (self.global_step % len(file_content) + 1), self.global_step) # Approx
                    # Log progress text
                    # Log progress text
                    self.writer.add_text("Status", f"Epoch {epoch_idx}: Step {self.global_step}, Loss: {batch_loss:.4f}", self.global_step)
                
                pbar.set_postfix({'loss': batch_loss})
        
        if self.scheduler:
            self.scheduler.step()
            
        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/TrainEpoch", avg_loss, epoch_idx)
        return avg_loss

    def validate(self, epoch_idx):
        """
        Runs validation on the validation set.
        Computes Loss, SHD, and F1.
        """
        self.model.eval()
        total_loss = 0.0
        total_shd = 0.0
        total_f1 = 0.0
        num_graphs = 0
        
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
                    
                    pred_delta, attn = self.model(x, mask, value)
                    
                    # Compute Delta Target
                    target_delta = target - x
                    loss = self.criterion(pred_delta, target_delta)
                    total_loss += loss.item()
                    
                    # Compute Metrics on the first batch of the SCM
                    if first_batch and 'adj' in mini_batch:
                        true_adj = mini_batch['adj'].numpy()
                        if true_adj.ndim == 3 and true_adj.shape[0] == 1:
                            true_adj = true_adj[0]
                            
                        # Extract predicted DAG from attention
                        # attn is (batch, N, N). Use the first sample or average?
                        # extract_attention_dag handles averaging/single sample
                        pred_adj = extract_attention_dag(attn, threshold=self.edge_threshold)
                        
                        # Debug shapes if mismatch
                        if pred_adj.shape != true_adj.shape:
                            # print(f"Shape mismatch! Pred: {pred_adj.shape}, True: {true_adj.shape}")
                            pass
                        
                        if pred_adj.shape == true_adj.shape:
                            shd = compute_shd(pred_adj, true_adj)
                            f1 = compute_f1(pred_adj, true_adj)
                            
                            total_shd += shd
                            total_f1 += f1
                            num_graphs += 1
                            
                            # Log Heatmap (only for the first graph of the epoch to avoid spam)
                            if num_graphs == 1:
                                fig = plot_adjacency_heatmap(pred_adj, true_adj)
                                self.writer.add_figure("Val/Adjacency", fig, epoch_idx)
                                plt.close(fig)
                            
                        first_batch = False
                    
        avg_loss = total_loss / len(self.val_loader)
        avg_shd = total_shd / max(1, num_graphs)
        avg_f1 = total_f1 / max(1, num_graphs)
        
        self.writer.add_scalar("Loss/Val", avg_loss, epoch_idx)
        self.writer.add_scalar("Metrics/SHD", avg_shd, epoch_idx)
        self.writer.add_scalar("Metrics/F1", avg_f1, epoch_idx)
        
        return avg_loss, avg_shd, avg_f1
