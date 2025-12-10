#############
Core Library
#############

.. currentmodule:: jax_privacy


Public API
----------

.. toctree::
   :maxdepth: 1
   :hidden:

   api/jax_privacy/jax_privacy.batch_selection
   api/jax_privacy/jax_privacy.clipping
   api/jax_privacy/jax_privacy.noise_addition
   api/jax_privacy/jax_privacy.auditing

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`api/jax_privacy/jax_privacy.batch_selection`
     - API and implementations for batch selection strategies.
   * - :doc:`api/jax_privacy/jax_privacy.clipping`
     - Utilities for clipping function outputs and aggregating across a batch.
   * - :doc:`api/jax_privacy/jax_privacy.noise_addition`
     - Implementations of optax.GradientTransformations that add noise to gradients.
   * - :doc:`api/jax_privacy/jax_privacy.auditing`
     - Library for empirical privacy auditing/estimation.


Experimental Modules
--------------------

.. toctree::
   :maxdepth: 1
   :hidden:

   api/jax_privacy/jax_privacy.experimental.execution_plan
   api/jax_privacy/jax_privacy.experimental.compilation_utils

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`api/jax_privacy/jax_privacy.experimental.execution_plan`
     - Module for defining DP Execution Plans.
   * - :doc:`api/jax_privacy/jax_privacy.experimental.compilation_utils`
     - Experimental utilities for handling variable batch sizes.


Other References
----------------

.. toctree::
   :maxdepth: 1
   :hidden:

   api/jax_privacy/jax_privacy.experimental.microbatching

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`api/jax_privacy/jax_privacy.experimental.microbatching`
     - A module for applying a function in a microbatched manner.
