from dataclasses import dataclass
from typing import Tuple, Sequence

@dataclass
class TransformerConfig:
    vocab_size: int = 1000
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_len: int = 32
    dropout_rate: float = 0.0

    @classmethod
    def small(cls):
        # ~1M parameters
        return cls(
            vocab_size=1000,
            hidden_size=256,
            num_heads=4,
            num_layers=2,
            max_len=64,
            dropout_rate=0.1
        )

    @classmethod
    def medium(cls):
        # ~10M parameters
        return cls(
            vocab_size=1000,
            hidden_size=384,
            num_heads=8,
            num_layers=6,
            max_len=128,
            dropout_rate=0.1
        )

    @classmethod
    def large(cls):
        # ~100M parameters
        return cls(
            vocab_size=30000,
            hidden_size=768,
            num_heads=12,
            num_layers=12,
            max_len=512,
            dropout_rate=0.1
        )

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
        # (32*3*3*3) + (64*32*3*3) + linear heads
        # 864 + 18432 ... mostly in linear layers if images are large
        # Let's adjust features/hidden size
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
