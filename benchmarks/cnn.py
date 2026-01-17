from flax import nnx
import jax.numpy as jnp
import jax
from typing import Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

@dataclass
class CNNConfig:
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    features: Sequence[int] = (32, 64)
    kernel_size: Tuple[int, int] = (3, 3)
    hidden_size: int = 128

    @classmethod
    def small(cls):
        # ~1M params
        return cls(
            input_shape=(32, 32, 3),
            features=(32, 64),
            hidden_size=256
        )

    @classmethod
    def medium(cls):
        # ~10M params
        return cls(
            input_shape=(64, 64, 3),
            features=(64, 128, 256),
            hidden_size=1024
        )

    @classmethod
    def large(cls):
        # ~100M params
        return cls(
            input_shape=(128, 128, 3),
            features=(64, 128, 256, 512, 512),
            hidden_size=4096
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

class CNN(nnx.Module):
    """A basic CNN model implemented with Flax NNX."""
    def __init__(self, config: CNNConfig, rngs: nnx.Rngs):
        self.config = config
        layers = []

        in_features = config.input_shape[-1]
        for features in config.features:
            layers.append(nnx.Conv(in_features, features, config.kernel_size, rngs=rngs))
            layers.append(nnx.relu)
            # nnx.max_pool is a functional, wrap it
            layers.append(lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
            in_features = features

        self.conv_layers = nnx.Sequential(*layers)

        # Calculate flattened size
        dummy_input = jnp.zeros((1,) + config.input_shape)
        out = self.conv_layers(dummy_input)
        flat_size = out.size

        self.head = nnx.Sequential(
            lambda x: x.reshape((x.shape[0], -1)),
            nnx.Linear(flat_size, config.hidden_size, rngs=rngs),
            nnx.relu,
            nnx.Linear(config.hidden_size, config.num_classes, rngs=rngs)
        )

    def __call__(self, x):
        x = self.conv_layers(x)
        logits = self.head(x)
        return logits

def generate_dummy_data(batch_size, input_shape, num_classes, seed=0):
    """Generates dummy images and labels for benchmarking using numpy."""
    np.random.seed(seed)
    # Generate data in NHWC format (standard for images)
    # We will let the consumer transpose if needed (e.g. Torch NCHW)
    # The config.input_shape is usually (H, W, C).

    # Actually, let's keep it simple. Flax expects (N, H, W, C).
    # Torch expects (N, C, H, W).
    # We will generate (N, H, W, C).

    images = np.random.randn(batch_size, *input_shape).astype(np.float32)
    labels = np.random.randint(0, num_classes, (batch_size,)).astype(np.int32)
    return images, labels

class CNNTorch(nn.Module):
    """A basic CNN model implemented with PyTorch."""
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        layers = []

        # Flax input is (H, W, C), but PyTorch uses (C, H, W).
        # We assume the config.input_shape is (H, W, C) as in Flax version.
        # So in_channels comes from the last dimension of input_shape.
        in_channels = config.input_shape[-1]

        # Replicating padding logic to match Flax (often SAME padding)
        padding = config.kernel_size[0] // 2

        for out_channels in config.features:
            layers.append(nn.Conv2d(in_channels, out_channels, config.kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate flattened size
        # Create a dummy input in NCHW format: (1, C, H, W)
        dummy_input = torch.zeros(1, config.input_shape[2], config.input_shape[0], config.input_shape[1])
        with torch.no_grad():
            out = self.conv_layers(dummy_input)
        flat_size = out.numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        logits = self.head(x)
        return logits
