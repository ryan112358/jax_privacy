# {py:mod}`jax_privacy.accounting.calibrate`

```{py:module} jax_privacy.accounting.calibrate
```

```{autodoc2-docstring} jax_privacy.accounting.calibrate
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`calibrate_num_updates <jax_privacy.accounting.calibrate.calibrate_num_updates>`
  - ```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_num_updates
    :summary:
    ```
* - {py:obj}`calibrate_noise_multiplier <jax_privacy.accounting.calibrate.calibrate_noise_multiplier>`
  - ```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_noise_multiplier
    :summary:
    ```
* - {py:obj}`calibrate_batch_size <jax_privacy.accounting.calibrate.calibrate_batch_size>`
  - ```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_batch_size
    :summary:
    ```
````

### API

````{py:function} calibrate_num_updates(*, target_epsilon: float, accountant: jax_privacy.accounting.analysis.DpTrainingAccountant, noise_multipliers: float | collections.abc.Sequence[tuple[int, float]], batch_sizes: int | collections.abc.Sequence[tuple[int, int]], num_samples: int, target_delta: float, examples_per_user: int | None = None, cycle_length: int | None = None, truncated_batch_size: int | None = None, initial_max_updates: int = 4, initial_min_updates: int = 1, tol: float = 0.1) -> int
:canonical: jax_privacy.accounting.calibrate.calibrate_num_updates

```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_num_updates
```
````

````{py:function} calibrate_noise_multiplier(*, target_epsilon: float, accountant: jax_privacy.accounting.analysis.DpTrainingAccountant, batch_sizes: int | collections.abc.Sequence[tuple[int, int]], num_updates: int, num_samples: int, target_delta: float, examples_per_user: int | None = None, cycle_length: int | None = None, truncated_batch_size: int | None = None, initial_max_noise: float = 1.0, initial_min_noise: float = 0.0, tol: float = 0.01) -> float
:canonical: jax_privacy.accounting.calibrate.calibrate_noise_multiplier

```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_noise_multiplier
```
````

````{py:function} calibrate_batch_size(*, target_epsilon: float, accountant: jax_privacy.accounting.analysis.DpTrainingAccountant, noise_multipliers: float | collections.abc.Sequence[tuple[int, float]], num_updates: int, num_samples: int, target_delta: float, examples_per_user: int | None = None, cycle_length: int | None = None, truncated_batch_size: int | None = None, initial_max_batch_size: int = 8, initial_min_batch_size: int = 1, tol: float = 0.01) -> int
:canonical: jax_privacy.accounting.calibrate.calibrate_batch_size

```{autodoc2-docstring} jax_privacy.accounting.calibrate.calibrate_batch_size
```
````
