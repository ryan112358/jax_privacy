import torch
import torch.nn as nn
from typing import Sequence, Tuple

class CNNConfig:
    """Configuration for the CNN model."""
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (32, 32, 3),
                 num_classes: int = 10,
                 features: Sequence[int] = (32, 64),
                 kernel_size: Tuple[int, int] = (3, 3),
                 hidden_size: int = 128):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.features = features
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size

class CNN(nn.Module):
    """A basic CNN model implemented with PyTorch."""
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        layers = []

        # Flax input is (H, W, C), but PyTorch uses (C, H, W).
        # We assume the config.input_shape is (H, W, C) as in Flax version.
        # So in_channels comes from the last dimension of input_shape.
        in_channels = config.input_shape[-1]

        # We need to compute the spatial size after convolutions/pooling to set the linear layer size.
        # We'll do a dummy forward pass or calculate it.
        # Let's verify standard padding. Flax defaults to 'SAME' in many contexts, but nnx.Conv might be 'VALID'.
        # However, typically simple CNN benchmarks use padding to keep dimensions or not.
        # If we use valid padding, dimensions shrink.
        # The Flax code:
        # layers.append(nnx.Conv(in_features, features, config.kernel_size, rngs=rngs))
        # layers.append(nnx.relu)
        # layers.append(lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2)))

        # We will assume 'SAME' padding (padding=1 for 3x3) to be safe/standard for such nets,
        # or we check if we can replicate valid.
        # Replicating valid padding (padding=0) is safer if we don't know Flax default behavior for sure,
        # but 'SAME' is more common in modern nets.
        # Wait, Flax linen Conv defaults to 'SAME'. nnx wraps linen. So likely 'SAME'.
        # For kernel=3, padding=1 gives SAME.

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

def generate_dummy_data(batch_size, input_shape, num_classes, seed=0):
    """Generates dummy images and labels for benchmarking.

    Args:
        batch_size: The batch size.
        input_shape: The shape of a single image (H, W, C).
        num_classes: The number of classes.
        seed: Random seed.

    Returns:
        images: Tensor of shape (batch_size, C, H, W)
        labels: Tensor of shape (batch_size,)
    """
    torch.manual_seed(seed)
    # PyTorch uses NCHW
    c, h, w = input_shape[2], input_shape[0], input_shape[1]
    images = torch.randn(batch_size, c, h, w)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels
