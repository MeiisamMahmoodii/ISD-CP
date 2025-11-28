import os
import sys
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime

def analyze_run(log_file):
    """
    Analyzes a single TensorBoard log file and returns best metrics.
    """
    ea = EventAccumulator(log_file)
    ea.Reload()
    
    try:
        tags = ea.Tags()['scalars']
    except Exception:
        # Handle empty or corrupted logs
        return {'valid': False, 'file': log_file, 'error': 'Could not read tags'}

    stats = {
        'file': log_file,
        'timestamp': os.path.getctime(log_file),
        'best_f1': -1.0,
        'best_shd': float('inf'),
        'best_loss': float('inf'),
        'best_epoch': -1,
        'valid': False
    }
    
    if 'Metrics/F1' in tags and 'Metrics/SHD' in tags:
        f1_events = ea.Scalars('Metrics/F1')
        shd_events = ea.Scalars('Metrics/SHD')
        
        f1_dict = {e.step: e.value for e in f1_events}
        shd_dict = {e.step: e.value for e in shd_events}
        
        common_steps = sorted(list(set(f1_dict.keys()) & set(shd_dict.keys())))
        
        if common_steps:
            f1_values = np.array([f1_dict[s] for s in common_steps])
            shd_values = np.array([shd_dict[s] for s in common_steps])
            steps = np.array(common_steps)
            
            # Find best F1
            max_f1 = np.max(f1_values)
            best_f1_indices = np.where(f1_values == max_f1)[0]
            
            # Tie break with SHD
            candidate_shds = shd_values[best_f1_indices]
            min_shd_idx_local = np.argmin(candidate_shds)
            best_idx = best_f1_indices[min_shd_idx_local]
            
            stats['best_f1'] = f1_values[best_idx]
            stats['best_shd'] = shd_values[best_idx]
            stats['best_epoch'] = steps[best_idx] + 1
            stats['valid'] = True
            
            # Get Loss if available for that step
            if 'Loss/Val' in tags:
                loss_events = ea.Scalars('Loss/Val')
                loss_dict = {e.step: e.value for e in loss_events}
                if steps[best_idx] in loss_dict:
                    stats['best_loss'] = loss_dict[steps[best_idx]]
    
    return stats

def compare_runs(log_dir):
    log_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not log_files:
        print("No log files found.")
        return

    print(f"Found {len(log_files)} runs. Analyzing...")
    
    results = []
    for f in log_files:
        try:
            stats = analyze_run(f)
            if stats['valid']:
                results.append(stats)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")
            
    if not results:
        print("No valid runs found (must have Metrics/F1 and Metrics/SHD).")
        return

    # Sort by timestamp
    results.sort(key=lambda x: x['timestamp'])
    
    print("\n" + "="*95)
    print(f"{'Run Date':<20} | {'Epoch':<5} | {'F1':<8} | {'SHD':<6} | {'Loss':<8} | {'Log File'}")
    print("-" * 95)
    
    best_overall = None
    
    for res in results:
        date_str = datetime.fromtimestamp(res['timestamp']).strftime('%Y-%m-%d %H:%M')
        fname = os.path.basename(res['file'])
        # Truncate fname if too long
        if len(fname) > 40:
            fname = fname[:37] + "..."
            
        print(f"{date_str:<20} | {res['best_epoch']:<5} | {res['best_f1']:.4f}   | {res['best_shd']:.2f}   | {res['best_loss']:.4f}   | {fname}")
        
        # Determine overall best
        if best_overall is None:
            best_overall = res
        else:
            # Logic: Higher F1 is better. If equal, Lower SHD is better.
            if res['best_f1'] > best_overall['best_f1']:
                best_overall = res
            elif res['best_f1'] == best_overall['best_f1']:
                if res['best_shd'] < best_overall['best_shd']:
                    best_overall = res
                    
    print("="*95)
    
    if best_overall:
        print(f"\n*** OVERALL BEST RUN ***")
        print(f"File: {best_overall['file']}")
        print(f"Run Date: {datetime.fromtimestamp(best_overall['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        print(f"Best Epoch: {best_overall['best_epoch']}")
        print(f"F1 Score: {best_overall['best_f1']:.4f}")
        print(f"SHD: {best_overall['best_shd']:.2f}")
        print(f"Loss: {best_overall['best_loss']:.4f}")
        print(f"Checkpoint File: model_epoch_{best_overall['best_epoch']}.pt")

if __name__ == "__main__":
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints_prod/logs'
    compare_runs(log_dir)
