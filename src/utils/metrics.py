import numpy as np
import torch
from sklearn.metrics import f1_score

def compute_shd(pred_adj: np.ndarray, true_adj: np.ndarray) -> int:
    """
    Computes Structural Hamming Distance (SHD).
    
    SHD counts the number of edge insertions, deletions, or flips required to transform
    one graph into another. It is a standard metric for evaluating causal discovery.
    
    Args:
        pred_adj: Predicted binary adjacency matrix (num_vars, num_vars).
        true_adj: True binary adjacency matrix.
        
    Returns:
        Integer representing the SHD.
    """
    diff = np.abs(pred_adj - true_adj)
    return np.sum(diff)

def compute_f1(pred_adj: np.ndarray, true_adj: np.ndarray) -> float:
    """
    Computes F1 Score for edge prediction.
    
    Args:
        pred_adj: Predicted binary adjacency matrix.
        true_adj: True binary adjacency matrix.
        
    Returns:
        F1 score (float).
    """
    y_true = true_adj.flatten()
    y_pred = pred_adj.flatten()
    return f1_score(y_true, y_pred, zero_division=0)

def extract_attention_dag(model, x, mask, value, threshold=0.1):
    """
    Extracts implicit DAG from attention weights.
    
    Args:
        model: The trained CausalTransformer model.
        x, mask, value: Input tensors.
        threshold: Value above which an attention weight is considered an edge.
        
    Returns:
        Adjacency matrix (numpy array).
    """
    # Get attention weights from the model
    # Shape: (batch_size, num_vars, num_vars)
    attn_weights = model.get_attention_maps(x, mask, value)
    
    # Average over batch
    avg_attn = attn_weights.mean(dim=0).numpy()
    
    # Thresholding
    # We ignore self-loops (diagonal) usually, but let's keep it simple.
    # A_{ij} = 1 if i causes j. 
    # Attention A_{ji} means j attends to i. So if j attends to i, i causes j.
    # So Adjacency = Attention.T
    
    adj = (avg_attn.T > threshold).astype(int)
    np.fill_diagonal(adj, 0) # Remove self-loops
    
    return adj
