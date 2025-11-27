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
    - **Supervised Attention**: It returns attention weights to be supervised by the ground truth DAG.
    
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
        self.num_layers = num_layers
        
        # 1. Embeddings
        # Feature Embedding: Project scalar feature to d_model using TabPFN style
        self.feature_emb = TabPFNStyleEmbedding(d_model)
        
        # Variable ID Embedding: Learnable vector for each variable
        # This allows the model to distinguish between "Temperature" and "Pressure"
        self.var_id_emb = nn.Embedding(num_vars, d_model)
        
        # Mask Embedding: Project binary mask to d_model
        # Tells the model: "This is the variable we are poking"
        self.mask_emb = nn.Linear(1, d_model)
        
        # Value Embedding: Project standardized intervention value to d_model
        # Tells the model: "We are setting it to this value (z-score)"
        self.value_emb = TabPFNStyleEmbedding(d_model)
        
        # 2. Transformer Encoder
        # We use a ModuleList of TransformerEncoderLayers to manually control the forward pass
        # and extract attention weights.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 3. Output Head
        # Predicts scalar Z-score for each variable
        self.output_head = nn.Linear(d_model, 1)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        initrange = 0.1
        self.var_id_emb.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)
        self.mask_emb.weight.data.uniform_(-initrange, initrange)
        self.mask_emb.bias.data.zero_()

    def forward(self, x, mask, value):
        """
        Forward pass of the model.
        
        Args:
            x: Baseline features (batch_size, num_vars).
            mask: Intervention mask (batch_size, num_vars) - 1.0 if intervened, 0.0 otherwise.
            value: Intervention value (batch_size, num_vars) - Non-zero only at intervened idx.
            
        Returns:
            prediction: Predicted post-intervention values (batch_size, num_vars).
            avg_attn: Averaged attention weights (batch_size, num_vars, num_vars).
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
        
        # Transformer Pass with Attention Extraction
        current_out = token
        layer_attns = []
        
        for layer in self.layers:
            # We need to call layer.self_attn manually to get weights
            # MultiheadAttention forward signature: query, key, value
            # For self-attention: src, src, src
            
            # We assume batch_first=True was passed to encoder layer
            # layer.self_attn returns (attn_output, attn_output_weights)
            # We need to pass need_weights=True
            
            # Note: nn.TransformerEncoderLayer implementation details:
            # x = src
            # if self.norm_first:
            #     x = x + self._sa_block(self.norm1(x))
            #     x = x + self._ff_block(self.norm2(x))
            # else:
            #     x = self.norm1(x + self._sa_block(x))
            #     x = self.norm2(x + self._ff_block(x))
            
            # _sa_block calls self.self_attn
            
            # To correctly replicate the layer logic AND get weights, we have to be careful.
            # It's safer to just rely on the fact that we want the weights from the *current* representation.
            # We can run self_attn separately to get weights, and then run the layer normally.
            # This adds a bit of compute overhead (running attention twice) but ensures correctness of the forward pass.
            
            # 1. Extract Attention Weights
            # We use the input to the layer (or normalized input if norm_first)
            # Default TransformerEncoderLayer is norm_first=False (Post-LN).
            # So input to attention is just `current_out`.
            
            with torch.no_grad(): # We don't need gradients for the extraction pass if we only use it for logging/aux loss target?
                # Actually we DO need gradients if we want to supervise the attention weights!
                # So we cannot use no_grad.
                pass

            # However, running it twice is inefficient and might cause issues if we optimize the "extraction" path
            # but the "forward" path uses a different computation graph (though sharing weights).
            
            # Better approach:
            # Manually implement the layer logic here.
            # Assuming Post-LN (default):
            # x = self.norm1(x + self._sa_block(x))
            # x = self.norm2(x + self._ff_block(x))
            
            src = current_out
            
            # Self Attention Block
            # attn_output, attn_weights = layer.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
            # But layer.self_attn expects (seq, batch, dim) if batch_first=False.
            # We set batch_first=True.
            
            attn_output, attn_weights = layer.self_attn(
                src, src, src,
                need_weights=True,
                average_attn_weights=True # Returns (batch, seq, seq)
            )
            
            # Dropout and Residual
            src = src + layer.dropout1(attn_output)
            src = layer.norm1(src)
            
            # Feed Forward Block
            # layer.linear1, layer.activation, layer.dropout, layer.linear2, layer.dropout2
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            src = src + layer.dropout2(src2)
            src = layer.norm2(src)
            
            current_out = src
            layer_attns.append(attn_weights)
            
        # Stack and average across layers
        # Shape: (num_layers, batch, seq, seq)
        all_attns = torch.stack(layer_attns)
        avg_attn = torch.mean(all_attns, dim=0) # Average over layers
        
        # Prediction
        # Map back from d_model to scalar output
        prediction = self.output_head(current_out).squeeze(-1)
        
        return prediction, avg_attn

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

