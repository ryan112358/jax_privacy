from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import jax.numpy as jnp
from flax import nnx
import jax

# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class CNNConfig:
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    features: Sequence[int] = (32, 64)
    kernel_size: Tuple[int, int] = (3, 3)
    hidden_size: int = 128
    dropout_rate: float = 0.0

    @classmethod
    def small(cls):
        return cls(
            input_shape=(32, 32, 3),
            features=(32, 64),
            hidden_size=256
        )

    @classmethod
    def medium(cls):
        return cls(
            input_shape=(64, 64, 3),
            features=(64, 128, 256),
            hidden_size=1024
        )

    @classmethod
    def large(cls):
        return cls(
            input_shape=(128, 128, 3),
            features=(64, 128, 256, 512, 512),
            hidden_size=4096
        )

# ==============================================================================
# JAX / FLAX IMPLEMENTATION
# ==============================================================================

class CNN(nnx.Module):
    def __init__(self, config: CNNConfig, rngs: nnx.Rngs):
        self.config = config
        self.layers = []

        in_features = config.input_shape[-1]

        # Convolutions
        for features in config.features:
            self.layers.append(nnx.Conv(in_features, features, kernel_size=config.kernel_size, rngs=rngs))
            self.layers.append(nnx.relu)
            self.layers.append(lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
            in_features = features

        self.layers.append(lambda x: x.reshape((x.shape[0], -1)))

        # Compute flattened size
        dummy_input = jnp.zeros((1, *config.input_shape))
        x = dummy_input
        for layer in self.layers[:-1]: # exclude flatten lambda for now to check shape
             pass

        # Rough calculation or running it
        # Let's just rely on dense layer to infer input shape if possible or calc manually
        # Flax NNX Dense doesn't infer input shape dynamically like Linen?
        # Actually it does if we don't specify in_features? No, NNX requires it usually.
        # But let's calculate it.

        h, w, _ = config.input_shape
        for _ in config.features:
            h //= 2
            w //= 2
        flat_size = h * w * config.features[-1]

        self.fc1 = nnx.Linear(flat_size, config.hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(config.hidden_size, config.num_classes, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def generate_dummy_data(batch_size, input_shape, num_classes, key):
    key_data, key_targets = jax.random.split(key)
    data = jax.random.normal(key_data, (batch_size, *input_shape))
    targets = jax.random.randint(key_targets, (batch_size,), 0, num_classes)
    return data, targets


# ==============================================================================
# PYTORCH IMPLEMENTATION
# ==============================================================================

try:
    import torch
    import torch.nn as nn

    class CNNTorch(nn.Module):
        def __init__(self, config: CNNConfig):
            super().__init__()
            self.config = config
            layers = []
            in_channels = config.input_shape[-1] # Shape is (H, W, C) in config, but Torch uses (C, H, W)

            # Torch Input: (B, C, H, W)
            # Config Input: (H, W, C)

            for features in config.features:
                layers.append(nn.Conv2d(in_channels, features, kernel_size=config.kernel_size, padding=1)) # padding=1 to match "same" approx if kernel=3
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_channels = features

            self.features = nn.Sequential(*layers)

            h, w = config.input_shape[0], config.input_shape[1]
            for _ in config.features:
                h //= 2
                w //= 2
            flat_size = h * w * config.features[-1]

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_size, config.num_classes)
            )

        def forward(self, x):
            # Expecting x in (B, C, H, W)
            x = self.features(x)
            x = self.classifier(x)
            return x

    def generate_dummy_data_torch(batch_size, input_shape, num_classes, seed=42):
        torch.manual_seed(seed)
        # Input shape in config is (H, W, C), torch needs (B, C, H, W)
        h, w, c = input_shape
        data = torch.randn(batch_size, c, h, w)
        targets = torch.randint(0, num_classes, (batch_size,))
        return data, targets

except ImportError:
    CNNTorch = None
    generate_dummy_data_torch = None
