import time
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

def test_ui():
    console = Console()
    
    # Mock Data
    epochs = 25 # Enough to trigger rolling window (limit 20)
    epoch_history_data = []
    best_val_loss = float('inf')
    best_epoch_idx = -1

    # Layout Setup (Copied from Trainer)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}")
    )
    
    metric_table = Table(show_header=True, header_style="bold magenta", expand=True)
    metric_table.add_column("Metric", style="dim")
    metric_table.add_column("Value", justify="right")
    metric_table.add_row("Epoch", "0/0")
    
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

    def fmt_stat_tuple(curr, prev):
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
            if ep == best_epoch_idx + 1: 
                ep_str = f"[bold yellow]➤ {ep_str}[/bold yellow]"
                
            # Format Stats
            train_str = fmt_stat_tuple(train_stats, prev_train)
            val_str = fmt_stat_tuple(val_stats, prev_val)
            
            new_table.add_row(ep_str, train_str, val_str)
            
            prev_train = train_stats
            prev_val = val_stats
            
        layout.children[3].update(Panel(new_table, title="History", expand=True))

    # Simulation Loop
    with Live(layout, console=console, refresh_per_second=4) as live:
        task = progress.add_task("Training...", total=epochs, info="Starting...")
        
        for epoch in range(epochs):
            time.sleep(0.1) # Simulate work
            
            # Mock Stats
            train_avg = 1.0 - (epoch * 0.02)
            val_avg = 1.0 - (epoch * 0.015) + (0.1 if epoch % 5 == 0 else 0) # Some noise
            
            epoch_history_data.append({
                'epoch': epoch + 1,
                'train': (train_avg-0.1, train_avg, train_avg+0.1),
                'val': (val_avg-0.1, val_avg, val_avg+0.1)
            })
            
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                best_epoch_idx = epoch
            
            update_history_view()
            progress.advance(task)

if __name__ == "__main__":
    test_ui()
