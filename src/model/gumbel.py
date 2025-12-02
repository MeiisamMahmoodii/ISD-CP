import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelAdjacency(nn.Module):
    """
    Learns a binary Adjacency Matrix using Gumbel-Sigmoid (Amortized).
    
    This module takes node embeddings and predicts the adjacency matrix.
    """
    def __init__(self, d_model, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        # Projections for Query and Key to predict structure
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
        # Learnable bias to control initial sparsity
        self.bias = nn.Parameter(torch.ones(1) * 5.0)
        
    def forward(self, x, hard=False, tau=None):
        """
        Args:
            x (Tensor): Node embeddings (Batch, Num_Vars, D_Model).
            hard (bool): If True, returns discrete 0/1 values.
            tau (float): Temperature.
            
        Returns:
            adj (Tensor): (Batch, Num_Vars, Num_Vars) Adjacency matrix.
        """
        if tau is None:
            tau = self.temperature
            
        # Compute logits via Attention mechanism
        # Q = W_q * x
        # K = W_k * x
        # Logits = Q @ K.T
        
        Q = self.query(x)
        K = self.key(x)
        
        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + self.bias
        
        # Debug: Print stats
        # if self.training: # Print always
        #     print(f"Logits Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}, Max: {logits.max().item():.4f}, Min: {logits.min().item():.4f}")
        #     probs = torch.sigmoid(logits)
        #     print(f"Probs Mean: {probs.mean().item():.4f}, >0.5: {(probs > 0.5).float().mean().item():.4f}")

        # Clamp logits for stability
        logits = torch.clamp(logits, min=-10.0, max=10.0)
            
        # Mask diagonal (no self-loops)
        mask = torch.eye(logits.shape[1], device=logits.device).bool().unsqueeze(0)
        logits = logits.masked_fill(mask, -1e9)
        
        if self.training:
            # Gumbel-Sigmoid Sampling
            # y = sigmoid((logits + gumbel_noise) / tau)
            adj = F.gumbel_softmax(
                torch.stack([torch.zeros_like(logits), logits], dim=-1), # (B, N, N, 2)
                tau=tau,
                hard=hard
            )[:, :, :, 1] # Take probability of class 1 (Edge)
        else:
            # Inference: Deterministic threshold
            adj = (torch.sigmoid(logits) > 0.5).float()
            
        return adj
