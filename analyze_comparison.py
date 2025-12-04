import json
import os
from rich.console import Console
from rich.table import Table

def load_history(dir_name):
    path = os.path.join(dir_name, "training_history.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    # Return the last run (most recent)
    return data[-1] if data else None

dirs = {
    "Huber": "checkpoints_compare_huber",
    "Causal Focus": "checkpoints_compare_focus",
    "Three-Tier": "checkpoints_compare_tier"
}

console = Console()
table = Table(title="Loss Function Comparison Results", show_header=True, header_style="bold magenta")
table.add_column("Loss Function", style="cyan")
table.add_column("Best Val Loss", justify="right")
table.add_column("Final Val Loss", justify="right")
table.add_column("Final Train Loss", justify="right")

results = []

for name, d in dirs.items():
    res = load_history(d)
    if res:
        results.append((name, res))
        table.add_row(
            name,
            f"{res.get('best_val_loss', 999):.4f}",
            f"{res.get('final_val_loss', 999):.4f}",
            f"{res.get('final_train_loss', 999):.4f}"
        )
    else:
        table.add_row(name, "N/A", "N/A", "N/A")

console.print(table)

# Determine Winner
if results:
    # Sort by Best Val Loss
    results.sort(key=lambda x: x[1].get('best_val_loss', 999))
    winner = results[0][0]
    console.print(f"\n[bold green]Winner based on Validation Loss: {winner}[/bold green]")
