######################
Core Library (New API)
######################

.. currentmodule:: jax_privacy


Public API
----------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:

  batch_selection.BallsInBinsSampling
  batch_selection.BatchSelectionStrategy
  batch_selection.CyclicPoissonSampling
  batch_selection.UserSelectionStrategy
  batch_selection.pad_to_multiple_of
  batch_selection.split_and_pad_global_batch
  clipped_grad
  noise_addition.SupportedStrategies
  noise_addition.gaussian_privatizer
  noise_addition.matrix_factorization_privatizer


Experimental Modules
--------------------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:

  experimental.execution_plan
  experimental.compilation_utils


Other References
----------------
.. autosummary::
  :toctree: _autosummary_output
  :nosignatures:

  experimental.microbatching
