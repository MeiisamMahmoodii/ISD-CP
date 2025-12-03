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
import io
import os
import json
import psutil
import shutil
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group
from rich.text import Text


class CausalFocusLoss(nn.Module):
    def __init__(self, focus_weight=10.0, delta=1.0):
        super().__init__()
        self.focus_weight = focus_weight
        self.huber = nn.HuberLoss(reduction='none', delta=delta)

    def forward(self, pred, target, mask):
        """
        pred:   (Batch, N)
        target: (Batch, N)
        mask:   (Batch, N) -> 1.0 for intervened node, 0.0 for others
        """
        # 1. Calculate standard element-wise loss
        raw_loss = self.huber(pred, target)
        
        # 2. Create a weight map
        # If mask=1 (Intervened Node), weight = focus_weight
        # If mask=0 (Other Nodes), weight = 1.0
        weights = 1.0 + (self.focus_weight - 1.0) * mask
        
        # 3. Apply weights
        weighted_loss = (raw_loss * weights).mean()
        
        return weighted_loss

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
        epochs: int = 100, # Total epochs for annealing
        loss_fn_name: str = "huber"
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
        
        if loss_fn_name == "causal_focus":
            self.criterion = CausalFocusLoss(focus_weight=10.0, delta=1.0)
            print("Using CausalFocusLoss (weight=10.0)")
        else:
            self.criterion = nn.HuberLoss(delta=1.0)
            print("Using Standard HuberLoss")
            self.criterion = nn.HuberLoss(delta=1.0)
            print("Using Standard HuberLoss")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.console = Console()
        self.output_dir = os.path.dirname(log_dir) # Assuming log_dir is output_dir/logs
        self.gpu_total_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

    def create_gauge_renderable(self, percentage, text_value, width=15):
        """Returns a Table grid with a gauge on the left and text on the right."""
        # Clamp percentage
        percentage = max(0, min(100, percentage))
        
        # Determine color
        if percentage < 60:
            color = "green"
        elif percentage < 85:
            color = "yellow"
        else:
            color = "red"
            
        num_blocks = int(round(percentage / 100 * width))
        bar = "█" * num_blocks + "░" * (width - num_blocks)
        bar_str = f"[{color}]{bar}[/{color}]"
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_row(bar_str, text_value)
        
        return grid

    def get_system_metrics(self):
        metrics = {}
        
        # CPU & RAM
        cpu_pct = psutil.cpu_percent()
        metrics['cpu'] = self.create_gauge_renderable(cpu_pct, f"{cpu_pct:.1f}%")
        
        ram_pct = psutil.virtual_memory().percent
        metrics['ram'] = self.create_gauge_renderable(ram_pct, f"{ram_pct:.1f}%")
        
        # GPU & VRAM
        if torch.cuda.is_available():
            # VRAM
            reserved = torch.cuda.memory_reserved(0)
            total = self.gpu_total_mem
            vram_pct = (reserved / total) * 100
            metrics['vram'] = self.create_gauge_renderable(vram_pct, f"{reserved / 1e9:.1f}/{total / 1e9:.1f} GB")
            
            # GPU Util
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                gpu_util = float(result.stdout.strip())
                metrics['gpu'] = self.create_gauge_renderable(gpu_util, f"{gpu_util:.1f}%")
            except:
                metrics['gpu'] = "N/A"
        else:
            metrics['vram'] = "N/A"
            metrics['gpu'] = "N/A"
            
        return metrics

    def fit(self):
        """
        Main training loop with persistent dashboard.
        """
        # Layout Setup
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}")
        )
        
        # Tasks
        train_task = progress.add_task("Training", total=len(self.train_loader), info="Waiting...", visible=False)
        val_task = progress.add_task("Validation", total=len(self.val_loader), info="Waiting...", visible=False)
        
        # Metrics Table (Current Step)
        metric_table = Table(show_header=True, header_style="bold magenta", expand=True)
        metric_table.add_column("Metric", style="dim")
        metric_table.add_column("Value", justify="right")
        metric_table.add_row("Epoch", "0/0")
        metric_table.add_row("Loss", "0.00")
        metric_table.add_row("GradNorm", "0.00")
        metric_table.add_row("LR", "0.00")
        metric_table.add_row("GPU", "0%")
        metric_table.add_row("VRAM", "0/0 GB")
        metric_table.add_row("RAM", "0%")
        metric_table.add_row("CPU", "0%")

        # Best Result Panel
        best_result_panel = Panel(
            "Waiting for results...",
            title="Best Result",
            style="bold green",
            expand=True
        )

        # History Table (Epoch Results)
        history_table = Table(show_header=True, header_style="bold cyan", expand=True, title="Epoch History (Last 20)")
        history_table.add_column("Epoch", justify="right")
        history_table.add_column("Train (Min/Avg/Max)", justify="center")
        history_table.add_column("Val (Min/Avg/Max)", justify="center")
        
        # Layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(progress, title="Progress", expand=True), size=6),
            Layout(Panel(metric_table, title="Current Metrics", expand=True), size=16),
            Layout(best_result_panel, size=3),
            Layout(Panel(history_table, title="History", expand=True))
        )
        
        best_val_loss = float('inf')
        best_epoch_idx = -1
        
        # Store history data for rebuilding table
        # List of dicts: {'epoch': int, 'train': (min, avg, max), 'val': (min, avg, max)}
        epoch_history_data = []

        def fmt_stat_tuple(curr, prev):
            # curr and prev are tuples (min, avg, max)
            # Returns formatted string "min / avg / max" with colors
            
            comps = []
            for i in range(3):
                val = curr[i]
                val_str = f"{val:.4f}"
                
                if prev:
                    prev_val = prev[i]
                    if val < prev_val:
                        val_str = f"[green]{val_str}[/green]"
                    elif val > prev_val:
                        val_str = f"[red]{val_str}[/red]"
                
                comps.append(val_str)
            
            return f"{comps[0]} / {comps[1]} / {comps[2]}"

        def update_history_view():
            # Update Best Result Panel
            if best_epoch_idx != -1:
                best_msg = f"Best Result: Epoch {best_epoch_idx + 1} | Val Loss: {best_val_loss:.4f}"
                layout.children[2].update(Panel(best_msg, title="Best Result", style="bold green", expand=True))

            # Rebuild table
            new_table = Table(show_header=True, header_style="bold cyan", expand=True, title="Epoch History (Last 20)")
            new_table.add_column("Epoch", justify="right")
            new_table.add_column("Train (Min/Avg/Max)", justify="center")
            new_table.add_column("Val (Min/Avg/Max)", justify="center")
            
            prev_train = None
            prev_val = None
            
            # Rolling Window: Show only last 20 epochs
            display_data = epoch_history_data[-20:]
            
            for entry in display_data:
                ep = entry['epoch']
                train_stats = entry['train']
                val_stats = entry['val']
                
                # Format Epoch with Arrow if Best
                ep_str = str(ep)
                if ep == best_epoch_idx + 1: # best_epoch_idx is 0-indexed
                    ep_str = f"[bold yellow]➤ {ep_str}[/bold yellow]"
                    
                # Format Stats
                train_str = fmt_stat_tuple(train_stats, prev_train)
                val_str = fmt_stat_tuple(val_stats, prev_val)
                
                new_table.add_row(ep_str, train_str, val_str)
                
                prev_train = train_stats
                prev_val = val_stats
                
            layout.children[3].update(Panel(new_table, title="History", expand=True))

        with Live(layout, console=self.console, refresh_per_second=4) as live:
            for epoch in range(self.epochs):
                # Update Dataset Epoch
                if hasattr(self.train_loader.dataset, 'set_epoch'):
                     self.train_loader.dataset.set_epoch(epoch) # Assuming reuse_factor=1 for now or handled inside
                
                # Reset Progress
                progress.reset(train_task)
                progress.update(train_task, visible=True, description=f"Epoch {epoch+1}/{self.epochs} [Train]")
                
                # Train
                train_avg, train_min, train_max = self.train_epoch(epoch, progress, train_task, metric_table, layout)
                
                # Validate
                progress.reset(val_task)
                progress.update(val_task, visible=True, description=f"Epoch {epoch+1}/{self.epochs} [Val]")
                val_avg, val_min, val_max = self.validate(epoch, progress, val_task)
                
                # Update History Data
                epoch_history_data.append({
                    'epoch': epoch + 1,
                    'train': (train_min, train_avg, train_max),
                    'val': (val_min, val_avg, val_max)
                })
                
                # Check Best
                if val_avg < best_val_loss:
                    best_val_loss = val_avg
                    best_epoch_idx = epoch
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model_best.pt"))
                
                # Update Table View
                update_history_view()
                
                # Checkpointing (Last)
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "model_last.pt"))
                    
                # Log to TensorBoard
                self.writer.add_scalar("Loss/Val_Best", best_val_loss, epoch)

        # --- End of Training Summary ---
        
        # 1. Compute Run Stats
        run_result = {
            "epochs": self.epochs,
            "best_val_loss": best_val_loss,
            "final_train_loss": train_avg, 
            "final_val_loss": val_avg 
        }
        
        # 2. Load/Update History
        history_file = os.path.join(self.output_dir, "training_history.json")
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    history = json.load(f)
            except:
                pass
                
        history.append(run_result)
        
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
            
        # 3. Display Comparison Table
        # We can print this to the console after the Live context exits
        table = Table(title="Training Run Summary", show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Metric", style="dim")
        table.add_column("Current Run", style="bold green")
        
        # Get Previous Run
        prev_run = history[-2] if len(history) >= 2 else None
        
        # Get Best Run (based on best_val_loss)
        best_run = None
        if history:
            # Filter out runs with inf loss
            valid_runs = [r for r in history if r.get('best_val_loss', float('inf')) != float('inf')]
            if valid_runs:
                best_run = min(valid_runs, key=lambda x: x.get('best_val_loss', float('inf')))

        if prev_run:
            table.add_column("Previous Run", style="yellow")
            table.add_column("Improvement", style="bold magenta")
        else:
            table.add_column("Previous Run", style="dim")
            table.add_column("Improvement", style="dim")
            
        if best_run:
            table.add_column("Best Run", style="cyan")
            
        # Metrics to compare
        metrics = [
            ("Best Val Loss", "best_val_loss", False),
            ("Final Val Loss", "final_val_loss", False),
            ("Final Train Loss", "final_train_loss", False)
        ]
        
        for label, key, higher_better in metrics:
            curr_val = run_result.get(key, 0.0)
            curr_str = f"{curr_val:.4f}"
            
            if prev_run:
                prev_val = prev_run.get(key, 0.0)
                
                # Calculate Improvement
                # For Loss: Lower is better. 
                # If curr < prev, diff should be positive (Improvement)
                raw_diff = curr_val - prev_val
                
                if not higher_better:
                    # Lower is better
                    # If curr (0.5) < prev (1.0) -> Improved by 0.5
                    # diff = -(0.5 - 1.0) = 0.5
                    diff = -raw_diff
                else:
                    # Higher is better
                    diff = raw_diff
                
                if diff > 0:
                    imp_str = f"[green]+{diff:.4f}[/green]"
                    curr_str = f"[green]{curr_str}[/green]"
                elif diff < 0:
                    imp_str = f"[red]{diff:.4f}[/red]"
                    curr_str = f"[red]{curr_str}[/red]"
                else:
                    imp_str = "="
                    
                row = [label, curr_str, f"{prev_val:.4f}", imp_str]
            else:
                row = [label, curr_str, "N/A", "N/A"]
            
            if best_run:
                best_val = best_run.get(key, 0.0)
                row.append(f"{best_val:.4f}")
            
            table.add_row(*row)
            
        self.console.print(Panel(table, title="Run Comparison", expand=False))
                
    def train_epoch(self, epoch_idx, progress, task_id, metric_table, layout):
        """
        Runs one full epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        total_batches_processed = 0
        min_loss = float('inf')
        max_loss = float('-inf')
        
        
        # No internal Live context anymore
        total_scms = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
                # Handle nested list structure from DataLoader
                file_content = batch
                if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], list):
                     file_content = batch[0]
                
                total_batches = len(file_content)
                for j, mini_batch in enumerate(file_content):
                    x = mini_batch['x'].to(self.device)
                    mask = mini_batch['mask'].to(self.device)
                    value = mini_batch['value'].to(self.device)
                    target = mini_batch['target'].to(self.device)
                    
                    if x.ndim == 3 and x.shape[0] == 1:
                        x = x.squeeze(0)
                        mask = mask.squeeze(0)
                        value = value.squeeze(0)
                        target = target.squeeze(0)
                    
                    # Micro-batching
                    batch_size = x.shape[0]
                    num_micro_batches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size
                    
                    batch_loss = 0.0
                    grad_norm = 0.0
                    self.optimizer.zero_grad()
                    
                    for micro_batch_idx in range(num_micro_batches):
                        start_idx = micro_batch_idx * self.micro_batch_size
                        end_idx = min((micro_batch_idx + 1) * self.micro_batch_size, batch_size)
                        
                        x_micro = x[start_idx:end_idx]
                        mask_micro = mask[start_idx:end_idx]
                        value_micro = value[start_idx:end_idx]
                        target_micro = target[start_idx:end_idx]
                        
                        pred_delta, adj = self.model(x_micro, mask_micro, value_micro)
                        target_delta = target_micro - x_micro
                        
                        if isinstance(self.criterion, CausalFocusLoss):
                            loss_pred = self.criterion(pred_delta, target_delta, mask_micro)
                        else:
                            loss_pred = self.criterion(pred_delta, target_delta)
                        
                        loss_micro = loss_pred
                        
                        if torch.isnan(loss_micro) or torch.isinf(loss_micro):
                            continue
                        
                        weight = (end_idx - start_idx) / batch_size
                        loss_scaled = loss_micro * weight / self.accumulation_steps
                        loss_scaled.backward()
                        
                        batch_loss += loss_micro.item() * weight
                    
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
                                    break
                        
                        if not has_nan:
                            self.optimizer.step()
                            if self.scheduler:
                                self.scheduler.step()
                        else:
                            print(f"Skipping step {self.global_step} due to NaN gradients.")
                            
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        
                        # Log to TensorBoard
                        if self.global_step % 50 == 0:
                            self.writer.add_scalar("Loss/Train", loss_micro.item(), self.global_step)
                            self.writer.add_scalar("Loss/Train_Pred", loss_pred.item(), self.global_step)
                            self.writer.add_scalar("System/LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                            
                            # Log progress text (Terminal Output in TensorBoard)
                            self.writer.add_text("Logs/Terminal", f"```\nEpoch {epoch_idx}: Step {self.global_step}, Loss: {batch_loss:.4f}\n```", self.global_step)
                        
                        # Log Distributions occasionally
                        if self.global_step % 200 == 0:
                            self.writer.add_histogram("Dist/Delta_Pred", pred_delta, self.global_step)
                            self.writer.add_histogram("Dist/Delta_True", target_delta, self.global_step)

                        # Update Live Display
                        progress.update(task_id, info=f"Loss: {batch_loss:.4f}")

                        # Update Table (Re-create rows)
                        metric_table = Table(show_header=True, header_style="bold magenta", expand=True)
                        metric_table.add_column("Metric", style="dim")
                        metric_table.add_column("Value", justify="right") # Keep right for scalars, Grids will expand
                        
                        sys_metrics = self.get_system_metrics()
                        
                        # Epoch Gauge
                        epoch_pct = (epoch_idx / self.epochs) * 100
                        epoch_renderable = self.create_gauge_renderable(epoch_pct, f"{epoch_idx}/{self.epochs}")
                        
                        # SCM Gauge (previously Step)
                        scm_pct = ((i + 1) / total_scms) * 100
                        scm_renderable = self.create_gauge_renderable(scm_pct, f"{i + 1}/{total_scms}")
                        
                        # Batch Gauge (Progress within SCM)
                        batch_pct = ((j + 1) / total_batches) * 100
                        batch_renderable = self.create_gauge_renderable(batch_pct, f"{j + 1}/{total_batches}")
                        
                        metric_table.add_row("Epoch", epoch_renderable)
                        metric_table.add_row("SCM", scm_renderable)
                        metric_table.add_row("Batch", batch_renderable)
                        metric_table.add_row("Loss", f"{batch_loss:.4f}")
                        metric_table.add_row("GradNorm", f"{grad_norm:.2f}")
                        metric_table.add_row("LR", f"{self.optimizer.param_groups[0]['lr']:.2e}")
                        metric_table.add_row("GPU", sys_metrics.get('gpu', 'N/A'))
                        metric_table.add_row("VRAM", sys_metrics.get('vram', 'N/A'))
                        metric_table.add_row("RAM", sys_metrics.get('ram', 'N/A'))
                        metric_table.add_row("CPU", sys_metrics.get('cpu', 'N/A'))
                        
                        # Update Metrics Panel (Index 1 in fit layout)
                        layout.children[1].update(Panel(metric_table, title="Current Metrics", expand=True))
                
                total_loss += batch_loss
                total_batches_processed += 1
                min_loss = min(min_loss, batch_loss)
                max_loss = max(max_loss, batch_loss)
                progress.advance(task_id)

                
        if self.scheduler:
            self.scheduler.step()
            
        avg_loss = total_loss / total_batches_processed if total_batches_processed > 0 else 0.0
        self.writer.add_scalar("Loss/TrainEpoch", avg_loss, epoch_idx)
        return avg_loss, min_loss, max_loss

    def log_terminal_message(self, message, step):
        """Logs a text message to TensorBoard."""
        self.writer.add_text("Logs/Terminal", f"```\n{message}\n```", step)

    def validate(self, epoch_idx, progress, task_id):
        """
        Runs validation on the validation set.
        Computes Loss only (No DAG metrics).
        """
        self.model.eval()
        total_loss = 0.0
        total_batches_processed = 0
        min_loss = float('inf')
        max_loss = float('-inf')
        
        num_graphs = 0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                progress.advance(task_id)

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
                    
                    if isinstance(self.criterion, CausalFocusLoss):
                         loss = self.criterion(pred_delta, target_delta, mask)
                    else:
                         loss = self.criterion(pred_delta, target_delta)
                         
                    total_loss += loss.item()
                    total_batches_processed += 1
                    min_loss = min(min_loss, loss.item())
                    max_loss = max(max_loss, loss.item())
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
                    
        avg_loss = total_loss / total_batches_processed if total_batches_processed > 0 else 0.0
        avg_mae = total_mae / total_batches_processed if total_batches_processed > 0 else 0.0
        
        self.writer.add_scalar("Loss/Val", avg_loss, epoch_idx)
        self.writer.add_scalar("Metrics/Val_MAE", avg_mae, epoch_idx)
        
        return avg_loss, min_loss, max_loss
