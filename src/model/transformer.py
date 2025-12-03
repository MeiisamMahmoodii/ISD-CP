import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TabPFNStyleEmbedding(nn.Module):
    """
    Embeds scalar values using a small MLP, similar to TabPFN's approach for numerical features.
    This allows the model to learn non-linear representations of scalar values.
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model * 2)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (..., 1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        return x



class CausalTransformer(nn.Module):
    """
    Implicit Structure Discovery - Causal Prediction (ISD-CP) with TabICL-style Encoder.
    
    This model uses an interleaved sequence of Feature and Value tokens:
    [Feature1, Value1, Feature2, Value2, ...]
    
    It does NOT learn an explicit adjacency matrix anymore. It relies on full
    attention to learn causal mechanisms.
    """
    def __init__(
        self, 
        num_vars: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        num_layers: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: int = 0.1
    ):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # 1. Embeddings
        # Feature Token: Just the ID
        self.var_id_emb = nn.Embedding(num_vars, d_model)
        
        # Value Token: Value + Type (Observed vs Intervened)
        self.value_emb = TabPFNStyleEmbedding(d_model)
        self.type_emb = nn.Embedding(2, d_model) # 0=Observed, 1=Intervened
        
        # 2. Transformer Encoder
        # Sequence length will be 2 * num_vars
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True,
                norm_first=True # Pre-Norm is usually better
            )
            for _ in range(num_layers)
        ])
        
        # 3. Output Head
        # We predict delta for each variable based on its Value Token output
        self.output_head = nn.Linear(d_model, 1)
        
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.var_id_emb.weight.data.uniform_(-initrange, initrange)
        self.type_emb.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, value):
        """
        Args:
            x: (batch, num_vars) - Baseline values
            mask: (batch, num_vars) - 1.0 if intervened, 0.0 if observed
            value: (batch, num_vars) - Intervention values (only valid where mask=1)
            
        Returns:
            delta_pred: (batch, num_vars)
            adj: None (Compatibility return)
        """
        batch_size, num_vars = x.shape
        device = x.device
        
        # 1. Prepare Inputs
        # Combine baseline and intervention values
        # If mask=1, use value. If mask=0, use x.
        final_values = x * (1 - mask) + value * mask
        
        # Type IDs: 0 for Observed, 1 for Intervened
        type_ids = mask.long()
        
        # Variable IDs: 0 to N-1
        var_ids = torch.arange(num_vars, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 2. Create Tokens
        # Feature Tokens: [Batch, N, D]
        f_tokens = self.var_id_emb(var_ids)
        
        # Value Tokens: [Batch, N, D]
        v_emb = self.value_emb(final_values.unsqueeze(-1))
        t_emb = self.type_emb(type_ids)
        v_tokens = v_emb + t_emb
        
        # 3. Interleave Tokens
        # We want: [F1, V1, F2, V2, ...]
        # Stack along new dim: [Batch, N, 2, D]
        # Then flatten: [Batch, 2*N, D]
        stacked = torch.stack([f_tokens, v_tokens], dim=2)
        tokens = stacked.flatten(1, 2)
        
        # 4. Transformer Pass
        # No mask needed (Full Attention)
        current_out = tokens
        
        for layer in self.layers:
            current_out = layer(current_out)
            
        # 5. Extract Outputs
        # We want predictions for the variables.
        # Should we use the output of Feature Token or Value Token?
        # Usually Value Token contains the "state" after processing.
        # Let's use Value Token outputs (indices 1, 3, 5...)
        # tokens shape: [F0, V0, F1, V1, ...]
        # V indices: 1, 3, 5...
        
        # Reshape back to [Batch, N, 2, D]
        out_reshaped = current_out.view(batch_size, num_vars, 2, self.d_model)
        
        # Take the Value token output (index 1)
        v_out = out_reshaped[:, :, 1, :] # [Batch, N, D]
        
        # 6. Prediction
        delta_pred = self.output_head(v_out).squeeze(-1)
        
        return delta_pred, None

if __name__ == "__main__":
    # Test
    model = CausalTransformer(num_vars=10, d_model=64, num_layers=2)
    x = torch.randn(5, 10)
    mask = torch.zeros(5, 10)
    mask[:, 0] = 1.0 # Intervene on var 0
    value = torch.zeros(5, 10)
    value[:, 0] = 2.0
    
    pred, attn = model(x, mask, value)
    print("Prediction shape:", pred.shape)
    print("Attention shape:", attn.shape)

