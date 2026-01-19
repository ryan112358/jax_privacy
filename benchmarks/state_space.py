from dataclasses import dataclass
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from flax import nnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

@dataclass
class StateSpaceConfig:
    vocab_size: int = 1000
    hidden_size: int = 64
    num_layers: int = 2
    max_len: int = 32
    dropout_rate: float = 0.0
    framework: str = "jax"

    @classmethod
    def small(cls):
        # Aligned with TransformerConfig.small
        return cls(vocab_size=1000, hidden_size=256, num_layers=2, max_len=256)

    @classmethod
    def medium(cls):
        # Aligned with TransformerConfig.medium
        return cls(vocab_size=1000, hidden_size=384, num_layers=6, max_len=256)

    @classmethod
    def large(cls):
        # Aligned with TransformerConfig.large
        return cls(vocab_size=30000, hidden_size=768, num_layers=12, max_len=256)

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
            return StateSpaceModel(self, rngs=rngs)
        elif self.framework == "torch":
            return StateSpaceModelTorch(self)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def generate_dummy_data(self, batch_size):
        data = np.random.randint(0, self.vocab_size, (batch_size, self.max_len)).astype(np.int32)
        targets = np.random.randint(0, self.vocab_size, (batch_size, self.max_len)).astype(np.int32)
        return data, targets

# --- Flax NNX Implementation ---

class S4Block(nnx.Module):
    def __init__(self, hidden_size: int, max_len: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(max_len,),
            feature_group_count=hidden_size, # Depthwise
            padding='VALID', # We will manually pad
            rngs=rngs
        )
        self.linear = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.activation = nnx.gelu
        self.kernel_size = max_len

    def __call__(self, x):
        # x: (batch, seq_len, hidden_size)
        residual = x
        x = self.norm(x)

        # Causal padding: pad left by kernel_size - 1
        pad_width = self.kernel_size - 1
        x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)))

        x = self.conv(x_padded)
        x = self.activation(x)
        x = self.linear(x)
        return x + residual

class StateSpaceModel(nnx.Module):
    def __init__(self, config: StateSpaceConfig, rngs: nnx.Rngs):
        self.config = config
        self.embed = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.layers = nnx.List([
            S4Block(config.hidden_size, config.max_len, rngs)
            for _ in range(config.num_layers)
        ])
        self.norm_final = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, rngs=rngs)

    def __call__(self, x):
        # x: (batch, seq_len)
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

# --- PyTorch Implementation ---

class S4BlockTorch(nn.Module):
    def __init__(self, hidden_size: int, max_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=max_len,
            groups=hidden_size, # Depthwise
            padding=0 # Manual padding
        )
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.max_len = max_len

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        residual = x
        x = self.norm(x)

        # Transpose for Conv1d: (batch, hidden_size, seq_len)
        x = x.transpose(1, 2)

        # Causal padding: pad left by kernel_size - 1
        x = F.pad(x, (self.max_len - 1, 0))

        x = self.conv(x)

        # Transpose back: (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)

        x = self.activation(x)
        x = self.linear(x)
        return x + residual

class StateSpaceModelTorch(nn.Module):
    def __init__(self, config: StateSpaceConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            S4BlockTorch(config.hidden_size, config.max_len)
            for _ in range(config.num_layers)
        ])
        self.norm_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
