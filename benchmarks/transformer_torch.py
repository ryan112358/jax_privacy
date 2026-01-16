import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    def __init__(self,
                 vocab_size: int = 1000,
                 hidden_size: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 max_len: int = 32,
                 dropout_rate: float = 0.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout_rate

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # q, k, v projections
        # Flax MHA uses separate projections usually, or one big one.
        # PyTorch MHA uses one big one for in_proj_weight if all same size.
        # We'll use separate for clarity and to avoid shape manipulation issues.
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x: (batch, seq, hidden)
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)

        # Scores: (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask is (L, L) with -inf or 0.
            # We need to broadcast it to (B, H, L, L).
            # If mask is 2D (L, L), it broadcasts correctly against (B, H, L, L) usually?
            # Yes, (L, L) broadcasts to (..., L, L).
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Output: (B, H, L, L) @ (B, H, L, D) -> (B, H, L, D)
        out = torch.matmul(attn, v)

        # Reassemble: (B, H, L, D) -> (B, L, H, D) -> (B, L, H*D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq, hidden)
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_len, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config.hidden_size, config.num_heads, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        self.norm_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        # Create causal mask
        # Torch MultiheadAttention expects attn_mask of shape (L, S) or (N*num_heads, L, S)
        # values: 0 for keep, -inf for discard.
        # Using triu to set upper triangle (future) to -inf.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

        # Opacus DPMultiheadAttention seems to have a bug/limitation where it assumes
        # query.size(0) is the target sequence length, but if batch_first=True, query is (N, L, E).
        # It checks mask size against query.size(0) which is batch size N.
        # Standard nn.MultiheadAttention handles batch_first=True by internally transposing or handling dimensions.
        # To avoid this issue with Opacus DPMultiheadAttention, we can ensure we pass inputs as (L, N, E)
        # if using Opacus, or just stick to batch_first=False.
        # However, we want to match faithful re-implementation of architecture.
        # Changing batch_first to False in the definition is cleaner.

        x = self.embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

def generate_dummy_data(batch_size, seq_len, vocab_size, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len))
