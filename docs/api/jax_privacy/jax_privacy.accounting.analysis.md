# {py:mod}`jax_privacy.accounting.analysis`

```{py:module} jax_privacy.accounting.analysis
```

```{autodoc2-docstring} jax_privacy.accounting.analysis
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SamplingMethod <jax_privacy.accounting.analysis.SamplingMethod>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.SamplingMethod
    :summary:
    ```
* - {py:obj}`DpParams <jax_privacy.accounting.analysis.DpParams>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams
    :summary:
    ```
* - {py:obj}`TrainingAccountant <jax_privacy.accounting.analysis.TrainingAccountant>`
  -
* - {py:obj}`DpTrainingAccountant <jax_privacy.accounting.analysis.DpTrainingAccountant>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant
    :summary:
    ```
* - {py:obj}`DpsgdTrainingAccountant <jax_privacy.accounting.analysis.DpsgdTrainingAccountant>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingAccountant
    :summary:
    ```
* - {py:obj}`DpsgdTrainingUserLevelAccountant <jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant
    :summary:
    ```
* - {py:obj}`SingleReleaseTrainingAccountant <jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant
    :summary:
    ```
* - {py:obj}`CachedExperimentAccountant <jax_privacy.accounting.analysis.CachedExperimentAccountant>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.CachedExperimentAccountant
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BatchingScaleSchedule <jax_privacy.accounting.analysis.BatchingScaleSchedule>`
  - ```{autodoc2-docstring} jax_privacy.accounting.analysis.BatchingScaleSchedule
    :summary:
    ```
````

### API

`````{py:class} SamplingMethod(*args, **kwds)
:canonical: jax_privacy.accounting.analysis.SamplingMethod

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} jax_privacy.accounting.analysis.SamplingMethod
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.SamplingMethod.__init__
```

````{py:attribute} POISSON
:canonical: jax_privacy.accounting.analysis.SamplingMethod.POISSON
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.accounting.analysis.SamplingMethod.POISSON
```

````

````{py:attribute} FIXED_BATCH_SIZE
:canonical: jax_privacy.accounting.analysis.SamplingMethod.FIXED_BATCH_SIZE
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.accounting.analysis.SamplingMethod.FIXED_BATCH_SIZE
```

````

`````

````{py:data} BatchingScaleSchedule
:canonical: jax_privacy.accounting.analysis.BatchingScaleSchedule
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.BatchingScaleSchedule
```

````

`````{py:class} DpParams
:canonical: jax_privacy.accounting.analysis.DpParams

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams
```

````{py:attribute} noise_multipliers
:canonical: jax_privacy.accounting.analysis.DpParams.noise_multipliers
:type: float | collections.abc.Sequence[tuple[int, float]] | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.noise_multipliers
```

````

````{py:attribute} num_samples
:canonical: jax_privacy.accounting.analysis.DpParams.num_samples
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.num_samples
```

````

````{py:attribute} delta
:canonical: jax_privacy.accounting.analysis.DpParams.delta
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.delta
```

````

````{py:attribute} batch_size
:canonical: jax_privacy.accounting.analysis.DpParams.batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.batch_size
```

````

````{py:attribute} batch_size_scale_schedule
:canonical: jax_privacy.accounting.analysis.DpParams.batch_size_scale_schedule
:type: jax_privacy.accounting.analysis.BatchingScaleSchedule
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.batch_size_scale_schedule
```

````

````{py:attribute} is_finite_guarantee
:canonical: jax_privacy.accounting.analysis.DpParams.is_finite_guarantee
:type: bool
:value: >
   True

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.is_finite_guarantee
```

````

````{py:attribute} batch_sizes
:canonical: jax_privacy.accounting.analysis.DpParams.batch_sizes
:type: collections.abc.Sequence[tuple[int, chex.Numeric]] | int
:value: >
   'field(...)'

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.batch_sizes
```

````

````{py:attribute} examples_per_user
:canonical: jax_privacy.accounting.analysis.DpParams.examples_per_user
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.examples_per_user
```

````

````{py:attribute} cycle_length
:canonical: jax_privacy.accounting.analysis.DpParams.cycle_length
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.cycle_length
```

````

````{py:attribute} sampling_method
:canonical: jax_privacy.accounting.analysis.DpParams.sampling_method
:type: jax_privacy.accounting.analysis.SamplingMethod
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.sampling_method
```

````

````{py:attribute} truncated_batch_size
:canonical: jax_privacy.accounting.analysis.DpParams.truncated_batch_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpParams.truncated_batch_size
```

````

`````

`````{py:class} TrainingAccountant
:canonical: jax_privacy.accounting.analysis.TrainingAccountant

Bases: {py:obj}`typing.Protocol`

````{py:method} compute_epsilon(num_updates: chex.Numeric, dp_params: jax_privacy.accounting.analysis.DpParams, allow_approximate_cache: bool = False) -> float
:canonical: jax_privacy.accounting.analysis.TrainingAccountant.compute_epsilon

```{autodoc2-docstring} jax_privacy.accounting.analysis.TrainingAccountant.compute_epsilon
```

````

`````

`````{py:class} DpTrainingAccountant(dp_accountant_config: jax_privacy.accounting.accountants.DpAccountantConfig)
:canonical: jax_privacy.accounting.analysis.DpTrainingAccountant

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant.__init__
```

````{py:method} can_calibrate_steps() -> bool
:canonical: jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_steps
:abstractmethod:

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_steps
```

````

````{py:method} can_calibrate_batch_size() -> bool
:canonical: jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_batch_size
:abstractmethod:

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_batch_size
```

````

````{py:method} can_calibrate_noise_multipliers() -> bool
:canonical: jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_noise_multipliers
:abstractmethod:

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant.can_calibrate_noise_multipliers
```

````

````{py:method} compute_epsilon(num_updates: chex.Numeric, dp_params: jax_privacy.accounting.analysis.DpParams, allow_approximate_cache: bool = False) -> float
:canonical: jax_privacy.accounting.analysis.DpTrainingAccountant.compute_epsilon

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpTrainingAccountant.compute_epsilon
```

````

`````

`````{py:class} DpsgdTrainingAccountant(dp_accountant_config: jax_privacy.accounting.accountants.DpAccountantConfig)
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingAccountant

Bases: {py:obj}`jax_privacy.accounting.analysis.DpTrainingAccountant`

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingAccountant
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingAccountant.__init__
```

````{py:method} can_calibrate_steps() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingAccountant.can_calibrate_steps

````

````{py:method} can_calibrate_batch_size() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingAccountant.can_calibrate_batch_size

````

````{py:method} can_calibrate_noise_multipliers() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingAccountant.can_calibrate_noise_multipliers

````

`````

`````{py:class} DpsgdTrainingUserLevelAccountant(dp_accountant_config: jax_privacy.accounting.accountants.DpAccountantConfig)
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant

Bases: {py:obj}`jax_privacy.accounting.analysis.DpTrainingAccountant`

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant.__init__
```

````{py:method} can_calibrate_steps() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant.can_calibrate_steps

````

````{py:method} can_calibrate_batch_size() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant.can_calibrate_batch_size

````

````{py:method} can_calibrate_noise_multipliers() -> bool
:canonical: jax_privacy.accounting.analysis.DpsgdTrainingUserLevelAccountant.can_calibrate_noise_multipliers

````

`````

`````{py:class} SingleReleaseTrainingAccountant(dp_accountant_config: jax_privacy.accounting.accountants.DpAccountantConfig)
:canonical: jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant

Bases: {py:obj}`jax_privacy.accounting.analysis.DpTrainingAccountant`

```{autodoc2-docstring} jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant.__init__
```

````{py:method} can_calibrate_steps() -> bool
:canonical: jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant.can_calibrate_steps

````

````{py:method} can_calibrate_batch_size() -> bool
:canonical: jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant.can_calibrate_batch_size

````

````{py:method} can_calibrate_noise_multipliers() -> bool
:canonical: jax_privacy.accounting.analysis.SingleReleaseTrainingAccountant.can_calibrate_noise_multipliers

````

`````

`````{py:class} CachedExperimentAccountant(training_accountant: jax_privacy.accounting.analysis.DpTrainingAccountant, max_num_updates: int, num_cached_points: int = 100)
:canonical: jax_privacy.accounting.analysis.CachedExperimentAccountant

```{autodoc2-docstring} jax_privacy.accounting.analysis.CachedExperimentAccountant
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.accounting.analysis.CachedExperimentAccountant.__init__
```

````{py:method} compute_epsilon(num_updates: chex.Numeric, dp_params: jax_privacy.accounting.analysis.DpParams, allow_approximate_cache: bool = False) -> float
:canonical: jax_privacy.accounting.analysis.CachedExperimentAccountant.compute_epsilon

```{autodoc2-docstring} jax_privacy.accounting.analysis.CachedExperimentAccountant.compute_epsilon
```

````

`````
