from flax import nnx
import jax.numpy as jnp
import jax
from typing import Sequence, Tuple
from benchmarks.config import CNNConfig

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

def generate_dummy_data(batch_size, input_shape, num_classes, key):
    """Generates dummy images and labels for benchmarking."""
    key_img, key_label = jax.random.split(key)
    images = jax.random.normal(key_img, (batch_size,) + input_shape)
    labels = jax.random.randint(key_label, (batch_size,), 0, num_classes)
    return images, labels
