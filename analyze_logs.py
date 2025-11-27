import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def analyze_logs(log_dir):
    # Find the latest tfevents file
    list_of_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not list_of_files:
        print("No log files found.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Analyzing log file: {latest_file}")
    
    ea = EventAccumulator(latest_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")
    
    for tag in tags:
        events = ea.Scalars(tag)
        values = [e.value for e in events]
        steps = [e.step for e in events]
        
        if len(values) == 0:
            continue
            
        print(f"\n--- {tag} ---")
        print(f"Count: {len(values)}")
        print(f"Min: {np.min(values):.4f}")
        print(f"Max: {np.max(values):.4f}")
        print(f"Mean: {np.mean(values):.4f}")
        print(f"Last 5 values: {[f'{v:.4f}' for v in values[-5:]]}")
        
        # Simple trend analysis
        if len(values) > 10:
            first_half = np.mean(values[:len(values)//2])
            second_half = np.mean(values[len(values)//2:])
            print(f"Trend (First Half -> Second Half): {first_half:.4f} -> {second_half:.4f}")

import sys

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_prod/logs'
    analyze_logs(log_dir)
