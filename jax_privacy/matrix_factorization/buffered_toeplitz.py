# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Buffered banded Toeplitz matrices."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from . import streaming_matrix


@dataclasses.dataclass(frozen=True)
class BufferedBandedToeplitz:
  """A StreamingMatrix wrapper that buffers inputs."""
  pass

__all__ = ["BufferedBandedToeplitz"]
