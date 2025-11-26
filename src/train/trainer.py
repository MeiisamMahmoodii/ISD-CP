import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from src.utils.metrics import compute_shd, compute_f1, extract_attention_dag, plot_adjacency_heatmap
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
        micro_batch_size: int = 20 # Default safe size
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.micro_batch_size = micro_batch_size
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        
    def train_epoch(self, epoch_idx):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        
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
                    pred_micro = self.model(x_micro, mask_micro, value_micro)
                    
                    # Compute Loss
                    loss_micro = self.criterion(pred_micro, target_micro)
                    
                    # Normalize loss for micro-batches AND accumulation steps
                    # We want the gradients to average out over the full batch
                    # loss.backward() accumulates gradients.
                    # If we split batch into N chunks, we should divide loss by N?
                    # Yes, because sum(gradients) should be equivalent to full batch gradient.
                    # Full batch loss = mean(per_sample_loss).
                    # Micro batch loss = mean(per_sample_loss_micro).
                    # We want to optimize for Full Batch Loss.
                    # Gradients from micro batch are d(Mean_Micro)/dw.
                    # We want d(Mean_Full)/dw = sum(d(Mean_Micro)/dw * (N_micro / N_total))
                    # So we weight by micro_batch_size / total_batch_size.
                    
                    weight = (end_idx - start_idx) / batch_size
                    loss_scaled = loss_micro * weight / self.accumulation_steps
                    
                    loss_scaled.backward()
                    
                    batch_loss += loss_micro.item() * weight
                
                # Optimization step (Gradient Accumulation across logical batches)
                if (self.global_step + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += batch_loss
                self.global_step += 1
                
                # Log batch loss occasionally
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("Loss/TrainBatch", batch_loss, self.global_step)
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
                
                # We only need to compute SHD/F1 once per SCM (chunk), not per mini-batch
                # But the mini-batches are just interventions on the SAME SCM.
                # So we can pick the first mini-batch to extract the DAG.
                
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
                    
                    pred = self.model(x, mask, value)
                    loss = self.criterion(pred, target)
                    total_loss += loss.item()
                    
                    # Compute Metrics on the first batch of the SCM
                    if first_batch and 'adj' in mini_batch:
                        true_adj = mini_batch['adj'].numpy()
                        # Handle squeeze if needed (it might have batch dim from collate)
                        if true_adj.ndim == 3 and true_adj.shape[0] == 1:
                            true_adj = true_adj[0]
                            
                        # Extract predicted DAG from attention
                        pred_adj = extract_attention_dag(self.model, x, mask, value)
                        
                        # Debug shapes if mismatch
                        if pred_adj.shape != true_adj.shape:
                            print(f"Shape mismatch! Pred: {pred_adj.shape}, True: {true_adj.shape}")
                            # Resize true_adj if it's smaller (likely due to networkx graph generation quirk?)
                            # Or maybe the graph has fewer nodes than num_vars if some are isolated?
                            # No, networkx.to_numpy_array should return (N, N) where N is number of nodes in G.
                            # If G has fewer nodes than num_vars, that's the issue.
                            # SCMGenerator initializes G with num_vars nodes.
                            
                            # Let's force shape match by padding or cropping (though cropping is bad).
                            # Better: Ensure SCMGenerator always creates graph with num_vars nodes.
                            pass
                        
                        if pred_adj.shape == true_adj.shape:
                            shd = compute_shd(pred_adj, true_adj)
                            f1 = compute_f1(pred_adj, true_adj)
                            
                            total_shd += shd
                            total_f1 += f1
                            num_graphs += 1
                            
                            # Log Heatmap
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
