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

"""Banded matrices (which are by definition sparse)."""

import functools
from typing import Any

import jax
import jax.numpy as jnp

from . import checks
from . import streaming_matrix


# Disabling pylint invalid-name to allow mathematical notation including
# single-capital-letter variables for matrices.
# See README.md for notation conventions.
# pylint:disable=invalid-name


def _lower_banded_init(value):
  return jnp.zeros_like(value)


def _lower_banded_next(
    value,
    state,
    bands: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes the next matrix-vector product for a lower banded matrix.

  Args:
    value: The next input value.
    state: A buffer of recent input values.
    bands: A [b] array representing the entries of a band.

  Returns:
    A tuple (output value, next state).
  """
  # We assume bands[0] is the diagonal, bands[1] is 1 below, etc.
  # So result[i] = sum_{j=0}^{b-1} bands[j] * input[i-j]
  # We assume state stores input[i-1], input[i-2], ...
  # so we can prepend value to state to get input[i], input[i-1], ...
  # and take the dot product.
  # Note: bands can be an array of scalars or an array of matrices (for block
  # matrices).
  # We assume value and state elements have the same shape.
  # If bands is 1D, we can use jnp.dot.
  # If bands is 3D (b, n, n), we need to do matrix-vector products.
  # Actually, StreamingMatrix is defined to work on PyTrees.
  # The implementation below assumes bands is simple (scalars).
  # If we need block matrices, we might need a more complex implementation.
  # For now, let's stick to the scalar case as used in Toeplitz.

  # Update state: push value, pop oldest.
  # State size should be b-1.
  # Actually, we need b inputs including current one.
  # If we store b-1 previous inputs, we prepend current value.
  window = jnp.concatenate([value[None], state])
  # Compute dot product.
  # bands shape: (b,)
  # window shape: (b, ...)
  # We want sum(bands[j] * window[j])
  # We can use tensordot or einsum.
  # If value is a scalar/array, and bands is 1D array.
  # bands[j] * window[j] broadcasts.
  # Sum over first axis.
  b = bands.shape[0]
  # Ensure window has size at least b (might need padding if starting up?)
  # The state is initialized to zeros, so it works.
  # window size is state size + 1.
  # We need state size to be b-1.
  result = jnp.tensordot(bands, window[:b], axes=(0, 0))
  # New state: window[:b-1] (which includes value, excludes oldest)
  # But wait, window[0] is current value (index i). window[1] is i-1.
  # bands[0] corresponds to diagonal (i, i).
  # bands[1] corresponds to (i, i-1).
  # So dot product aligns: bands[j] * window[j] is C_{i, i-j} * x_{i-j}.
  new_state = window[: b - 1]
  return result, new_state


def from_bands(bands: jax.Array) -> streaming_matrix.StreamingMatrix:
  """Creates a StreamingMatrix from a lower-triangular banded matrix.

  The matrix is defined by `bands`. `bands[0]` is the main diagonal.
  `bands[1]` is the first sub-diagonal, etc.
  The matrix is assumed to be Toeplitz (constant along diagonals) and
  lower triangular.

  Args:
    bands: A 1D array of shape (b,) representing the bands.

  Returns:
    A StreamingMatrix.
  """
  b = bands.shape[0]

  def init(value):
    # State needs to hold b-1 previous values.
    # Shape of state should be (b-1,) + value.shape
    return jnp.zeros((b - 1,) + value.shape, dtype=value.dtype)

  def next_step(value, state):
    return _lower_banded_next(value, state, bands)

  return streaming_matrix.StreamingMatrix.from_array_implementation(
      init, next_step
  )

__all__ = ["from_bands"]
