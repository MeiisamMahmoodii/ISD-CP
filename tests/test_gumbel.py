import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.gumbel import GumbelAdjacency
from src.model.transformer import CausalTransformer

def test_gumbel_adjacency():
    print("Testing GumbelAdjacency...")
    batch_size = 2
    num_vars = 5
    d_model = 16
    
    model = GumbelAdjacency(d_model=d_model)
    x = torch.randn(batch_size, num_vars, d_model)
    
    # Training mode (Soft)
    model.train()
    adj = model(x)
    print(f"Soft Adj shape: {adj.shape}")
    assert adj.shape == (batch_size, num_vars, num_vars)
    assert adj.min() >= 0 and adj.max() <= 1
    
    # Inference mode (Hard)
    model.eval()
    adj_hard = model(x)
    print(f"Hard Adj shape: {adj_hard.shape}")
    assert torch.all(torch.logical_or(adj_hard == 0, adj_hard == 1))
    
    print("GumbelAdjacency Passed!")

def test_causal_transformer_gumbel():
    print("Testing CausalTransformer with Gumbel...")
    batch_size = 2
    num_vars = 5
    d_model = 16
    
    model = CausalTransformer(num_vars=num_vars, d_model=d_model, num_layers=2)
    model.train()
    
    x = torch.randn(batch_size, num_vars)
    mask = torch.zeros(batch_size, num_vars)
    value = torch.zeros(batch_size, num_vars)
    
    # Forward pass
    pred, adj = model(x, mask, value)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Adjacency shape: {adj.shape}")
    
    assert pred.shape == (batch_size, num_vars)
    assert adj.shape == (batch_size, num_vars, num_vars)
    
    # Check gradients
    loss = pred.sum() + adj.sum()
    loss.backward()
    print("Backward pass successful!")
    
    print("CausalTransformer Gumbel Passed!")

if __name__ == "__main__":
    test_gumbel_adjacency()
    test_causal_transformer_gumbel()
