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

from src.model.gumbel import GumbelAdjacency

class CausalTransformer(nn.Module):
    """
    Structure-Agnostic Causal Transformer (ISD-CP) with Gumbel Attention.
    
    This model explicitly learns a binary Adjacency Matrix (Structure) and uses it
    to mask the attention in the Transformer (Function).
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
        self.nhead = nhead
        
        # 1. Embeddings
        self.feature_emb = TabPFNStyleEmbedding(d_model)
        self.var_id_emb = nn.Embedding(num_vars, d_model)
        self.mask_emb = nn.Linear(1, d_model)
        self.value_emb = TabPFNStyleEmbedding(d_model)
        
        # Sink Token
        self.sink_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. Structure Learner (Gumbel Adjacency)
        # Learns the N x N binary mask
        self.structure_learner = GumbelAdjacency(d_model)
        
        # 3. Transformer Encoder
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
        
        # 4. Output Head
        self.output_head = nn.Linear(d_model, 1)
        
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.var_id_emb.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)
        self.mask_emb.weight.data.uniform_(-initrange, initrange)
        self.mask_emb.bias.data.zero_()

    def forward(self, x, mask, value):
        """
        Args:
            x: (batch, num_vars)
            mask: (batch, num_vars)
            value: (batch, num_vars)
            
        Returns:
            delta_pred: (batch, num_vars)
            adj: (num_vars, num_vars) Learned Adjacency Matrix
        """
        batch_size, seq_len = x.shape
        
        # 1. Embeddings
        var_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x_emb = self.feature_emb(x.unsqueeze(-1))
        id_emb = self.var_id_emb(var_ids)
        m_emb = self.mask_emb(mask.unsqueeze(-1))
        v_emb = self.value_emb(value.unsqueeze(-1))
        
        token = x_emb + id_emb + m_emb + v_emb
        
        # 2. Learn Structure (Adjacency)
        # adj is (batch, num_vars, num_vars) with values in [0, 1]
        # We pass the embeddings to predict the structure (Amortized)
        adj = self.structure_learner(token, hard=not self.training) # Hard during inference
        
        # Prepend Sink Token
        sink = self.sink_token.expand(batch_size, -1, -1)
        token = torch.cat([sink, token], dim=1)
        
        # Create Attention Mask from Adjacency
        # We need a mask for (N+1, N+1)
        # Sink (0) attends to Sink (0)
        # Vars (1..N) attend to Sink (0) AND Parents (1..N)
        
        # adj is (Batch, N, N). adj[i, j]=1 means i->j.
        # Attention is (Target, Source).
        # Target j (index j+1) attends to Source i (index i+1) if adj[i, j]=1.
        
        # Base mask for Vars->Vars: (Batch, N, N)
        vars_mask = (adj.transpose(1, 2) == 0) # True = Block
        
        # Full mask: (Batch, N+1, N+1)
        # Initialize with False (Attend)
        full_mask = torch.zeros(batch_size, seq_len + 1, seq_len + 1, dtype=torch.bool, device=x.device)
        
        # 1. Sink -> Vars (Block)
        full_mask[:, 0, 1:] = True 
        
        # 2. Vars -> Sink (Allow - False) -> Already False
        
        # 3. Vars -> Vars (Copy from vars_mask)
        full_mask[:, 1:, 1:] = vars_mask
        
        # Expand for MultiheadAttention: (Batch * nhead, N+1, N+1)
        attn_mask = full_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        attn_mask = attn_mask.reshape(batch_size * self.nhead, seq_len + 1, seq_len + 1)
        
        # 3. Transformer Pass with Mask
        current_out = token
        
        for layer in self.layers:
            # We pass the same structural mask to all layers
            # layer(src, src_mask=...)
            # src_mask shape: (seq, seq) or (batch*num_heads, seq, seq)
            
            # Note: nn.TransformerEncoderLayer.forward(src, src_mask=...)
            # We need to convert boolean mask to float for some versions, or keep bool.
            # Newer PyTorch supports bool mask (True = Ignore).
            
            current_out = layer(current_out, src_mask=attn_mask)
            
        # 4. Prediction
        delta_pred = self.output_head(current_out[:, 1:, :]).squeeze(-1)
        
        return delta_pred, adj

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

