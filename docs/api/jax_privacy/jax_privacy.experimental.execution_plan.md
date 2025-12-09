# {py:mod}`jax_privacy.experimental.execution_plan`

```{py:module} jax_privacy.experimental.execution_plan
```

```{autodoc2-docstring} jax_privacy.experimental.execution_plan
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DPExecutionPlan <jax_privacy.experimental.execution_plan.DPExecutionPlan>`
  - ```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan
    :summary:
    ```
* - {py:obj}`BandMFExecutionPlanConfig <jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig>`
  - ```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NeighboringRelation <jax_privacy.experimental.execution_plan.NeighboringRelation>`
  - ```{autodoc2-docstring} jax_privacy.experimental.execution_plan.NeighboringRelation
    :summary:
    ```
````

### API

````{py:data} NeighboringRelation
:canonical: jax_privacy.experimental.execution_plan.NeighboringRelation
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.NeighboringRelation
```

````

`````{py:class} DPExecutionPlan
:canonical: jax_privacy.experimental.execution_plan.DPExecutionPlan

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan
```

````{py:attribute} clipped_aggregation_fn
:canonical: jax_privacy.experimental.execution_plan.DPExecutionPlan.clipped_aggregation_fn
:type: jax_privacy.clipping.BoundedSensitivityCallable
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan.clipped_aggregation_fn
```

````

````{py:attribute} batch_selection_strategy
:canonical: jax_privacy.experimental.execution_plan.DPExecutionPlan.batch_selection_strategy
:type: jax_privacy.batch_selection.BatchSelectionStrategy
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan.batch_selection_strategy
```

````

````{py:attribute} noise_addition_transform
:canonical: jax_privacy.experimental.execution_plan.DPExecutionPlan.noise_addition_transform
:type: optax.GradientTransformation
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan.noise_addition_transform
```

````

````{py:attribute} dp_event
:canonical: jax_privacy.experimental.execution_plan.DPExecutionPlan.dp_event
:type: dp_accounting.DpEvent
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.DPExecutionPlan.dp_event
```

````

`````

`````{py:class} BandMFExecutionPlanConfig
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig
```

````{py:attribute} iterations
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.iterations
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.iterations
```

````

````{py:attribute} num_bands
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.num_bands
:type: int
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.num_bands
```

````

````{py:attribute} epsilon
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.epsilon
:type: float | None
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.epsilon
```

````

````{py:attribute} delta
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.delta
:type: float | None
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.delta
```

````

````{py:attribute} noise_multiplier
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.noise_multiplier
:type: float | None
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.noise_multiplier
```

````

````{py:attribute} sampling_prob
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.sampling_prob
:type: float
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.sampling_prob
```

````

````{py:attribute} truncated_batch_size
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.truncated_batch_size
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.truncated_batch_size
```

````

````{py:attribute} num_examples
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.num_examples
:type: int | None
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.num_examples
```

````

````{py:attribute} partition_type
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.partition_type
:type: jax_privacy.batch_selection.PartitionType
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.partition_type
```

````

````{py:attribute} strategy_optimization_steps
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.strategy_optimization_steps
:type: int
:value: >
   500

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.strategy_optimization_steps
```

````

````{py:attribute} accountant
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.accountant
:type: dp_accounting.PrivacyAccountant
:value: >
   'Field(...)'

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.accountant
```

````

````{py:attribute} neighboring_relation
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.neighboring_relation
:type: dp_accounting.NeighboringRelation
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.neighboring_relation
```

````

````{py:attribute} noise_seed
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.noise_seed
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.noise_seed
```

````

````{py:method} make(clipped_aggregation_fn: jax_privacy.clipping.BoundedSensitivityCallable) -> jax_privacy.experimental.execution_plan.DPExecutionPlan
:canonical: jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.make

```{autodoc2-docstring} jax_privacy.experimental.execution_plan.BandMFExecutionPlanConfig.make
```

````

`````
