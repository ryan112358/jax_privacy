from flax import nnx
import jax.numpy as jnp
import jax
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    vocab_size: int = 1000
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_len: int = 256
    framework: str = "jax"

    @classmethod
    def small(cls):
        # ~1M parameters
        return cls(
            vocab_size=1000,
            hidden_size=256,
            num_heads=4,
            num_layers=2,
            max_len=256,
        )

    @classmethod
    def medium(cls):
        # ~10M parameters
        return cls(
            vocab_size=1000,
            hidden_size=384,
            num_heads=8,
            num_layers=6,
            max_len=256,
        )

    @classmethod
    def large(cls):
        # ~100M parameters
        return cls(
            vocab_size=30000,
            hidden_size=768,
            num_heads=12,
            num_layers=12,
            max_len=256,
        )

    @classmethod
    def build(cls, size):
        if size == 'small':
            return cls.small()
        elif size == 'medium':
            return cls.medium()
        elif size == 'large':
            return cls.large()
        else:
            raise ValueError(f"Unknown size: {size}")

    def make(self, rngs=None):
        if self.framework == "jax":
            return Transformer(self, rngs=rngs)
        elif self.framework == "torch":
            return TransformerTorch(self)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def generate_dummy_data(self, batch_size):
        data = np.random.randint(0, self.vocab_size, (batch_size, self.max_len)).astype(np.int32)
        targets = np.random.randint(0, self.vocab_size, (batch_size, self.max_len)).astype(np.int32)
        return data, targets

def causal_flash_attention_fn(query, key, value, **kwargs):
    return jax.nn.dot_product_attention(
        query.astype(jnp.bfloat16),
        key.astype(jnp.bfloat16),
        value.astype(jnp.bfloat16),
        implementation='xla' if jax.default_backend() == 'cpu' else 'cudnn',
        is_causal=True,
    ).astype(jnp.float32)

class TransformerBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            decode=False,
            rngs=rngs,
            attention_fn=causal_flash_attention_fn
        )
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, hidden_size * 4, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_size * 4, hidden_size, rngs=rngs)
        )
        # Removed Dropout as requested

    def __call__(self, x, mask=None):
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, mask=mask) # self-attention
        x = x + residual

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

class Transformer(nnx.Module):
    def __init__(self, config: TransformerConfig, rngs: nnx.Rngs):
        self.config = config
        self.embed = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.pos_embed = nnx.Embed(config.max_len, config.hidden_size, rngs=rngs)
        self.layers = nnx.List(
            [
                TransformerBlock(config.hidden_size, config.num_heads, rngs)
                for _ in range(config.num_layers)
            ]
        )
        self.norm_final = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, rngs=rngs)

    def __call__(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        pos = jnp.arange(seq_len)
        pos = jnp.broadcast_to(pos, (batch_size, seq_len))

        # Create masks
        mask = nnx.make_attention_mask(jnp.ones((batch_size, seq_len)), jnp.ones((batch_size, seq_len)))
        causal_mask = nnx.make_causal_mask(jnp.ones((batch_size, seq_len)))
        mask = mask + causal_mask

        x = self.embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # q, k, v projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (batch, seq, hidden)
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D)

        # Flash Attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reassemble: (B, H, L, D) -> (B, L, H, D) -> (B, L, H*D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.out_proj(out)

class TransformerBlockTorch(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # x: (batch, seq, hidden)
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

class TransformerTorch(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_len, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlockTorch(config.hidden_size, config.num_heads)
            for _ in range(config.num_layers)
        ])
        self.norm_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        device = x.device

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
