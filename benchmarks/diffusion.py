from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

# Config
@dataclass
class DiffusionConfig:
    image_size: int = 32
    channels: int = 3
    hidden_size: int = 64
    num_layers: int = 2
    framework: str = "jax"

    @classmethod
    def small(cls):
        return cls(hidden_size=64, num_layers=2)

    @classmethod
    def medium(cls):
        return cls(hidden_size=128, num_layers=4)

    @classmethod
    def large(cls):
        return cls(hidden_size=256, num_layers=6)

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
            return FlaxDiffusion(self, rngs=rngs)
        elif self.framework == "torch":
            return TorchDiffusion(self)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def generate_dummy_data(self, batch_size):
        # Returns ((x, t), target) as numpy arrays
        if self.framework == "jax":
             x = np.random.randn(batch_size, self.image_size, self.image_size, self.channels).astype(np.float32)
        elif self.framework == "torch":
             x = np.random.randn(batch_size, self.channels, self.image_size, self.image_size).astype(np.float32)
        else:
             raise ValueError(f"Unknown framework: {self.framework}")

        t = np.random.randint(0, 1000, (batch_size,)).astype(np.int32)
        target = np.random.randn(*x.shape).astype(np.float32)
        return (x, t), target

# Flax Implementation
class FlaxDiffusionBlock(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.norm = nnx.LayerNorm(channels, rngs=rngs)

    def __call__(self, x):
        h = self.conv1(x)
        h = nnx.gelu(h)
        h = self.conv2(h)
        h = self.norm(h + x)
        return h

class FlaxDiffusion(nnx.Module):
    def __init__(self, config: DiffusionConfig, rngs: nnx.Rngs):
        self.config = config
        self.input_proj = nnx.Conv(config.channels, config.hidden_size, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.time_embed = nnx.Embed(1000, config.hidden_size, rngs=rngs)

        self.layers = nnx.List([
            FlaxDiffusionBlock(config.hidden_size, rngs) for _ in range(config.num_layers)
        ])

        self.output_proj = nnx.Conv(config.hidden_size, config.channels, kernel_size=(3, 3), padding=1, rngs=rngs)

    def __call__(self, x, t):
        # x: (B, H, W, C)
        # t: (B,)
        t_emb = self.time_embed(t) # (B, C)
        t_emb = t_emb[:, None, None, :] # (B, 1, 1, C)

        h = self.input_proj(x)
        h = h + t_emb

        for layer in self.layers:
            h = layer(h)

        return self.output_proj(h)

def generate_dummy_data_flax(batch_size, config, key):
    # Returns (x, t), target
    # x: (B, H, W, C)
    # t: (B,)
    # target: (B, H, W, C)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (batch_size, config.image_size, config.image_size, config.channels))
    t = jax.random.randint(k2, (batch_size,), 0, 1000)
    target = jax.random.normal(k3, x.shape)
    return (x, t), target

# Torch Implementation
try:
    import torch
    import torch.nn as nn

    class TorchDiffusionBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            # Use GroupNorm instead of LayerNorm for 2D inputs in Torch, or BatchNorm
            self.norm = nn.GroupNorm(8, channels) # 8 groups

        def forward(self, x):
            h = self.conv1(x)
            h = nn.functional.gelu(h)
            h = self.conv2(h)
            h = self.norm(h + x)
            return h

    class TorchDiffusion(nn.Module):
        def __init__(self, config: DiffusionConfig):
            super().__init__()
            self.config = config
            self.input_proj = nn.Conv2d(config.channels, config.hidden_size, kernel_size=3, padding=1)
            self.time_embed = nn.Embedding(1000, config.hidden_size)

            self.layers = nn.ModuleList([
                TorchDiffusionBlock(config.hidden_size) for _ in range(config.num_layers)
            ])
            self.output_proj = nn.Conv2d(config.hidden_size, config.channels, kernel_size=3, padding=1)

        def forward(self, x, t):
            # x: (B, C, H, W)
            # t: (B,)
            t_emb = self.time_embed(t) # (B, C)
            t_emb = t_emb[:, :, None, None] # (B, C, 1, 1)

            h = self.input_proj(x)
            h = h + t_emb

            for layer in self.layers:
                h = layer(h)

            return self.output_proj(h)

    def generate_dummy_data_torch(batch_size, config, seed=42):
        torch.manual_seed(seed)
        # NCHW format
        x = torch.randn(batch_size, config.channels, config.image_size, config.image_size)
        t = torch.randint(0, 1000, (batch_size,))
        target = torch.randn_like(x)
        return (x, t), target

except ImportError:
    class TorchDiffusion:
        def __init__(self, *args, **kwargs):
            raise ImportError("Torch is not installed")

    def generate_dummy_data_torch(*args, **kwargs):
        raise ImportError("Torch is not installed")
