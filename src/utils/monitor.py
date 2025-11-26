import torch
import subprocess

def log_gpu_usage():
    """
    Logs current GPU memory usage and utilization.
    
    Uses torch.cuda for memory stats and nvidia-smi for utilization.
    This is critical for monitoring deep models to prevent OOM errors.
    """
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Optional: nvidia-smi
        try:
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
            print(f"NVIDIA-SMI: {result.decode('utf-8').strip()}")
        except:
            pass
    else:
        print("GPU not available.")
