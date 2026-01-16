from flax import nnx
import jax.numpy as jnp
import jax
from typing import Optional

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

class TransformerBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            decode=False,
            rngs=rngs
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
                TransformerBlock(config.hidden_size, config.num_heads, config.dropout_rate, rngs)
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

def generate_dummy_data(batch_size, seq_len, vocab_size, key):
    # Use int32 for compatibility with jax.value_and_grad in clipped_grad context if applicable,
    # though clipped_grad differentiates w.r.t. params not inputs.
    return jax.random.randint(key, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
