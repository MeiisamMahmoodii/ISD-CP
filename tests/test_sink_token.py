import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.transformer import CausalTransformer

def test_sink_token():
    print("Testing Sink Token Implementation...")
    
    batch_size = 2
    num_vars = 5
    d_model = 16
    
    model = CausalTransformer(num_vars=num_vars, d_model=d_model, num_layers=2)
    model.eval()
    
    x = torch.randn(batch_size, num_vars)
    mask = torch.zeros(batch_size, num_vars)
    value = torch.zeros(batch_size, num_vars)
    
    # Forward pass
    pred, attn = model(x, mask, value)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Check shapes
    assert pred.shape == (batch_size, num_vars), f"Expected pred shape {(batch_size, num_vars)}, got {pred.shape}"
    assert attn.shape == (batch_size, num_vars + 1, num_vars + 1), f"Expected attn shape {(batch_size, num_vars + 1, num_vars + 1)}, got {attn.shape}"
    
    # Check Softmax property (rows sum to 1)
    # attn is (Batch, Target, Source)
    # Sum over Source (last dim) should be 1
    row_sums = attn.sum(dim=-1)
    print(f"Row sums (first sample): {row_sums[0]}")
    
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention weights do not sum to 1!"
    
    print("Sink Token Test Passed!")

if __name__ == "__main__":
    test_sink_token()
