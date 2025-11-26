import torch
import torch.nn as nn
import math

class CausalTransformer(nn.Module):
    """
    Structure-Agnostic Causal Transformer (ISD-CP).
    
    This model predicts the consequences of an intervention on a causal system.
    It treats the problem as an end-to-end regression task:
    Input: Baseline State + Intervention Description
    Output: Post-Intervention State
    
    Key Innovation:
    - It does not require an explicit DAG.
    - It uses self-attention to implicitly learn the causal structure (dependencies) 
      between variables.
    - It uses a "Fixed Reference Frame" via standardized inputs.
    
    Input Token Construction:
    For each variable i, the input token is a sum of:
    1. FeatureEmb: Embedding of the baseline value (S_baseline).
    2. VarID: Learnable embedding unique to variable i.
    3. Mask: Embedding indicating if this variable is the one being intervened on.
    4. Value: Embedding of the intervention magnitude (z-score).
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
        """
        Args:
            num_vars: Number of variables in the system.
            d_model: Dimension of the transformer embeddings.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of the feedforward network inside transformer.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # 1. Embeddings
        # Feature Embedding: Project scalar feature to d_model
        self.feature_emb = nn.Linear(1, d_model)
        
        # Variable ID Embedding: Learnable vector for each variable
        # This allows the model to distinguish between "Temperature" and "Pressure"
        self.var_id_emb = nn.Embedding(num_vars, d_model)
        
        # Mask Embedding: Project binary mask to d_model
        # Tells the model: "This is the variable we are poking"
        self.mask_emb = nn.Linear(1, d_model)
        
        # Value Embedding: Project standardized intervention value to d_model
        # Tells the model: "We are setting it to this value (z-score)"
        self.value_emb = nn.Linear(1, d_model)
        
        # 2. Transformer Encoder
        # Standard PyTorch Transformer Encoder
        # Bidirectional attention allows all variables to attend to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Head
        # Predicts scalar Z-score for each variable
        self.output_head = nn.Linear(d_model, 1)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        initrange = 0.1
        self.feature_emb.weight.data.uniform_(-initrange, initrange)
        self.feature_emb.bias.data.zero_()
        self.var_id_emb.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, mask, value):
        """
        Forward pass of the model.
        
        Args:
            x: Baseline features (batch_size, num_vars).
            mask: Intervention mask (batch_size, num_vars) - 1.0 if intervened, 0.0 otherwise.
            value: Intervention value (batch_size, num_vars) - Non-zero only at intervened idx.
            
        Returns:
            Predicted post-intervention values (batch_size, num_vars).
        """
        batch_size, seq_len = x.shape
        
        # Create VarIDs (0 to num_vars-1)
        var_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        # Reshape inputs to (batch, seq, 1) for Linear layers
        x_emb = self.feature_emb(x.unsqueeze(-1))
        id_emb = self.var_id_emb(var_ids)
        m_emb = self.mask_emb(mask.unsqueeze(-1))
        v_emb = self.value_emb(value.unsqueeze(-1))
        
        # Combine embeddings
        # Token = Feature + VarID + Mask + Value
        # This summation fuses all information into a single vector per variable
        token = x_emb + id_emb + m_emb + v_emb
        
        # Transformer Pass
        # No causal mask (bidirectional) - we want to learn the full graph structure
        output = self.transformer_encoder(token)
        
        # Prediction
        # Map back from d_model to scalar output
        prediction = self.output_head(output).squeeze(-1)
        
        return prediction

    def get_attention_maps(self, x, mask, value):
        """
        Runs a forward pass and returns the averaged attention weights across all heads and layers.
        
        Args:
            x, mask, value: Input tensors.
            
        Returns:
            Attention matrix (batch_size, num_vars, num_vars).
        """
        self.eval()
        attentions = []
        
        def hook_fn(module, input, output):
            # output is (attn_output, attn_output_weights) if need_weights=True
            # But nn.TransformerEncoderLayer calls self_attn(..., need_weights=False) by default usually?
            # Actually, in PyTorch implementation of TransformerEncoderLayer:
            # x, _ = self.self_attn(x, x, x, key_padding_mask=...)
            # It discards weights.
            # So hooks on the module output won't work if weights aren't returned.
            pass

        # Since TransformerEncoderLayer doesn't return weights, we can't easily hook it 
        # unless we rely on the internal implementation details or use a custom encoder.
        
        # ALTERNATIVE: Re-implement the encoder loop manually for this method.
        # This ensures we can force need_weights=True.
        
        batch_size, seq_len = x.shape
        var_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        x_emb = self.feature_emb(x.unsqueeze(-1))
        id_emb = self.var_id_emb(var_ids)
        m_emb = self.mask_emb(mask.unsqueeze(-1))
        v_emb = self.value_emb(value.unsqueeze(-1))
        
        src = x_emb + id_emb + m_emb + v_emb
        
        # Iterate through layers manually
        for layer in self.transformer_encoder.layers:
            # We access the self_attn module directly
            # layer.self_attn is a MultiheadAttention
            # We need to replicate what the layer does:
            #   x = src
            #   x = self.norm1(x + self._sa_block(x))
            #   x = self.norm2(x + self._ff_block(x))
            
            # But _sa_block calls self_attn.
            # We can call self_attn directly with need_weights=True
            
            # Note: This is brittle if PyTorch changes internal names, but standard for research code.
            # Standard TransformerEncoderLayer structure:
            # self.self_attn(src, src, src, ...)
            
            # Let's just call the attention module to get weights, 
            # but we must ensure we pass the right inputs (which is the output of previous layer).
            
            # To get the weights *used* during the pass, we really should have used a custom layer.
            # But for "extraction" (post-hoc), we can just run the attention again?
            # No, because the input to layer i depends on layer i-1.
            
            # Robust approach: Run the full layer, but capture weights via a temporary hook 
            # IF we can force need_weights=True. We can't easily force it on the standard layer.
            
            # FALLBACK: Use a custom TransformerEncoder implementation in this file 
            # that saves attention weights.
            pass
            
        # Let's replace the standard TransformerEncoder with a custom one that supports weight extraction.
        # For now, to minimize code changes, I will implement a custom forward pass 
        # that iterates layers and calls self_attn manually with need_weights=True.
        
        current_out = src
        layer_attns = []
        
        for layer in self.transformer_encoder.layers:
            # Replicate Pre-Norm or Post-Norm? PyTorch default is Post-Norm.
            # x = norm(x + attn(x))
            
            # 1. Self Attention
            # We need to call layer.self_attn manually to get weights
            # MultiheadAttention forward signature: query, key, value
            # For self-attention: src, src, src
            
            # We assume batch_first=True was passed to encoder layer
            attn_out, attn_weights = layer.self_attn(
                current_out, current_out, current_out,
                need_weights=True,
                average_attn_weights=True # Return (batch, seq, seq)
            )
            
            layer_attns.append(attn_weights.detach().cpu())
            
            # Continue the forward pass to get input for next layer
            # We can just call the layer! 
            # But we already computed attn_out.
            # Let's just call layer(current_out) to be safe and correct, 
            # even if it recomputes attention (without weights).
            # It's slightly inefficient but correct.
            current_out = layer(current_out)
            
        # Stack and average across layers
        # Shape: (num_layers, batch, seq, seq)
        all_attns = torch.stack(layer_attns)
        avg_attn = torch.mean(all_attns, dim=0) # Average over layers
        
        return avg_attn

if __name__ == "__main__":
    # Test
    model = CausalTransformer(num_vars=10, d_model=64, num_layers=2)
    x = torch.randn(5, 10)
    mask = torch.zeros(5, 10)
    mask[:, 0] = 1.0 # Intervene on var 0
    value = torch.zeros(5, 10)
    value[:, 0] = 2.0
    
    out = model(x, mask, value)
    print("Output shape:", out.shape)
