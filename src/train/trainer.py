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
                
                # Anneal Gumbel Temperature
                # 1.0 -> 0.1 over epochs
                if hasattr(self.model, 'structure_learner') and hasattr(self.model.structure_learner, 'temperature'):
                    # Exponential decay: tau = tau_start * (tau_end / tau_start) ^ (epoch / max_epochs)
                    tau_start = 1.0
                    tau_end = 0.1
                    progress = epoch_idx / self.epochs
                    new_tau = tau_start * (tau_end / tau_start) ** progress
                    self.model.structure_learner.temperature = new_tau
                    
                    self.writer.add_scalar("System/Gumbel_Tau", new_tau, epoch_idx)

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
                    
                    # Compute Structure Loss (Supervised Adjacency)
                    loss_aux = torch.tensor(0.0, device=self.device)
                    if true_adj is not None:
                        if true_adj.ndim == 2:
                            target_adj = true_adj.unsqueeze(0).expand(adj.shape[0], -1, -1)
                        else:
                            target_adj = true_adj[start_idx:end_idx]
                        
                        # adj is (Batch, N, N) - Wait, model returns adj from GumbelAdjacency
                        # GumbelAdjacency returns (Batch, N, N).
                        # The Sink Token is added in Transformer Forward, but NOT to the adjacency matrix itself.
                        # The Transformer uses the Sink Token for masking, but the Gumbel module only predicts N x N.
                        # So 'adj' returned by model is still (Batch, N, N).
                        # So we don't need to slice it!
                        
                        # BCE Loss for binary edges
                        loss_aux = F.binary_cross_entropy(adj, target_adj)
                    
                    # Compute Sparsity Penalty (L1 on Adjacency)
                    loss_sparse = torch.mean(torch.abs(adj))

                    # Compute DAG Constraint (Trace Exponential)
                    # h(A) = tr(e^A) - d = 0
                    # We need to ensure A has zero diagonal for this to work well, or handle it.
                    # GumbelAdjacency already masks diagonal to 0.÷
                    # We compute this per sample and average? Or average A then compute?
                    # Usually computed on the "average" A or per sample.
                    # Since A is sample-dependent here (contextual), we compute per sample.
                    
                    # Matrix Exp is expensive. We can use polynomial approximation or just run it.
                    # For small graphs (20 vars), it's fast.
                    
                    # adj is (Batch, N, N)
                    # torch.matrix_exp supports batches
                    if adj.shape[1] <= 50: # Only run for small graphs to avoid OOM/Slow
                        expm_A = torch.matrix_exp(adj)
                        h_A = torch.diagonal(expm_A, dim1=-2, dim2=-1).sum(-1) - adj.shape[1]
                        loss_dag = torch.mean(h_A * h_A) # Quadratic penalty
                    else:
                        loss_dag = torch.tensor(0.0, device=self.device)

                    # Total Loss
                    # We add loss_dag with a coefficient (rho) which we can increase over time?
                    # For now, just a fixed weight.
                    lambda_dag = 0.1 
                    
                    loss_micro = loss_pred + self.lambda_aux * loss_aux + self.lambda_sparse * loss_sparse + lambda_dag * loss_dag
                    
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
                        self.writer.add_scalar("Train/Loss", loss_micro.item(), self.global_step)
                        self.writer.add_scalar("Train/Pred_Loss", loss_pred.item(), self.global_step)
                        self.writer.add_scalar("Train/Aux_Loss", loss_aux.item(), self.global_step)
                        self.writer.add_scalar("Train/Sparse_Loss", loss_sparse.item(), self.global_step)
                        self.writer.add_scalar("System/GradNorm", grad_norm, self.global_step)
                        self.writer.add_scalar("System/LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                        
                        # Compute Metrics for Progress Bar (on first sample of batch)
                        with torch.no_grad():
                            # Edges
                            probs = torch.sigmoid(adj)
                            num_edges = (probs > 0.5).float().sum(dim=(1, 2)).mean().item()
                            self.writer.add_scalar("Train/Avg_Edge_Count", num_edges, self.global_step)
                            
                            # SHD/F1 (CPU intensive, do only for one sample)
                            if true_adj is not None:
                                # true_adj is (N, N) for the current SCM
                                t_adj = true_adj.cpu().numpy()
                                p_adj = (probs[0] > 0.5).float().cpu().numpy()
                                
                                shd = compute_shd(p_adj, t_adj)
                                f1 = compute_f1(p_adj, t_adj)
                                
                                self.writer.add_scalar("Train/SHD", shd, self.global_step)
                                self.writer.add_scalar("Train/F1", f1, self.global_step)
                                
                                # Update persistent metrics
                                postfix_dict.update({
                                    'shd': f"{shd}",
                                    'f1': f"{f1:.2f}",
                                    'edges': f"{num_edges:.1f}",
                                    'grad': f"{grad_norm:.1f}",
                                    'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
                                })
                            else:
                                postfix_dict.update({
                                    'edges': f"{num_edges:.1f}",
                                    'grad': f"{grad_norm:.1f}"
                                })

            total_loss += batch_loss
            self.global_step += 1
            
            # Always update loss in postfix_dict
            postfix_dict['loss'] = f"{batch_loss:.2f}"
            pbar.set_postfix(postfix_dict)
            
            # Log batch loss occasionally
            if self.global_step % 50 == 0:
                # Log progress text (Terminal Output in TensorBoard)
                self.writer.add_text("Logs/Terminal", f"Epoch {epoch_idx}: Step {self.global_step}, Loss: {batch_loss:.4f}", self.global_step)
            
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
        self.writer.add_text("Logs/Terminal", message, step)

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
                    
                    pred_delta, adj = self.model(x, mask, value)
                    
                    # Compute Delta Target
                    target_delta = target - x
                    loss = self.criterion(pred_delta, target_delta)
                    total_loss += loss.item()
                    
                    # Compute Metrics on the first batch of the SCM
                    if first_batch and 'adj' in mini_batch:
                        true_adj = mini_batch['adj'].numpy()
                        if true_adj.ndim == 3 and true_adj.shape[0] == 1:
                            true_adj = true_adj[0]
                            
                        # Extract predicted DAG from Gumbel Adjacency
                        # adj is (batch, N, N). Use the first sample or average?
                        # Since GumbelAdjacency is contextual (input dependent? No, it's global parameters in this implementation!)
                        # Wait, my implementation of GumbelAdjacency has `self.logits` as a Parameter.
                        # So it learns ONE global graph?
                        # The user's problem is "Structure Discovery" which usually implies input-dependent if it's "Amortized".
                        # BUT, the `OnlineCausalDataset` generates NEW SCMs every epoch.
                        # If `GumbelAdjacency` learns ONE graph, it will fail because the graph changes every sample!
                        
                        # CRITICAL REALIZATION:
                        # The current `GumbelAdjacency` learns a STATIC graph.
                        # But the dataset provides DYNAMIC graphs (different for each sample).
                        # The model needs to PREDICT the graph from the input `x`.
                        
                        # I need to change `GumbelAdjacency` to be INPUT-DEPENDENT (Amortized).
                        # It should take `x` (embeddings) and predict `adj`.
                        
                        # Let's finish this replace first, then fix the model.
                        
                        pred_adj_soft = adj[0].detach().cpu().numpy()
                        pred_adj = (pred_adj_soft > 0.5).astype(int)
                        
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
                            
                            # Log Visualizations (only for the first graph of the epoch)
                            if num_graphs == 1:
                                # 1. Adjacency Heatmap
                                fig = plot_adjacency_heatmap(pred_adj, true_adj)
                                self.writer.add_figure("Analysis/Adjacency", fig, epoch_idx)
                                plt.close(fig)
                                
                                # 2. Scatter Plot (Pred vs True Delta)
                                fig2, ax = plt.subplots(figsize=(6, 6))
                                # Flatten and sample to avoid huge plots if large
                                t_flat = target_delta.cpu().numpy().flatten()
                                p_flat = pred_delta.cpu().numpy().flatten()
                                if len(t_flat) > 1000:
                                    idx = np.random.choice(len(t_flat), 1000, replace=False)
                                    t_flat = t_flat[idx]
                                    p_flat = p_flat[idx]
                                    
                                ax.scatter(t_flat, p_flat, alpha=0.3)
                                ax.plot([t_flat.min(), t_flat.max()], [t_flat.min(), t_flat.max()], 'r--') # Identity line
                                ax.set_xlabel("True Delta")
                                ax.set_ylabel("Predicted Delta")
                                ax.set_title(f"Prediction Correlation (Epoch {epoch_idx})")
                                ax.grid(True)
                                self.writer.add_figure("Analysis/Pred_vs_True", fig2, epoch_idx)
                                plt.close(fig2)
                                
                                # 3. Histograms
                                if torch.isnan(pred_delta).any() or torch.isinf(pred_delta).any():
                                    print(f"WARNING: NaNs/Infs detected in pred_delta at epoch {epoch_idx}!")
                                    print(f"pred_delta range: [{pred_delta.min()}, {pred_delta.max()}]")
                                else:
                                    self.writer.add_histogram("Dist/Val_Delta_Pred", pred_delta, epoch_idx)
                                    self.writer.add_histogram("Dist/Val_Delta_True", target_delta, epoch_idx)
                            
                        first_batch = False
                    
        avg_loss = total_loss / len(self.val_loader)
        avg_shd = total_shd / max(1, num_graphs)
        avg_f1 = total_f1 / max(1, num_graphs)
        
        self.writer.add_scalar("Loss/Val", avg_loss, epoch_idx)
        self.writer.add_scalar("Metrics/SHD", avg_shd, epoch_idx)
        self.writer.add_scalar("Metrics/F1", avg_f1, epoch_idx)
        
        return avg_loss, avg_shd, avg_f1
