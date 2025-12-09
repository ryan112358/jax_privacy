# {py:mod}`jax_privacy.keras_api`

```{py:module} jax_privacy.keras_api
```

```{autodoc2-docstring} jax_privacy.keras_api
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DPKerasConfig <jax_privacy.keras_api.DPKerasConfig>`
  - ```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_private <jax_privacy.keras_api.make_private>`
  - ```{autodoc2-docstring} jax_privacy.keras_api.make_private
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LossFn <jax_privacy.keras_api.LossFn>`
  - ```{autodoc2-docstring} jax_privacy.keras_api.LossFn
    :summary:
    ```
````

### API

`````{py:class} DPKerasConfig
:canonical: jax_privacy.keras_api.DPKerasConfig

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig
```

````{py:attribute} epsilon
:canonical: jax_privacy.keras_api.DPKerasConfig.epsilon
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.epsilon
```

````

````{py:attribute} delta
:canonical: jax_privacy.keras_api.DPKerasConfig.delta
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.delta
```

````

````{py:attribute} clipping_norm
:canonical: jax_privacy.keras_api.DPKerasConfig.clipping_norm
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.clipping_norm
```

````

````{py:attribute} batch_size
:canonical: jax_privacy.keras_api.DPKerasConfig.batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.batch_size
```

````

````{py:attribute} gradient_accumulation_steps
:canonical: jax_privacy.keras_api.DPKerasConfig.gradient_accumulation_steps
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.gradient_accumulation_steps
```

````

````{py:attribute} train_steps
:canonical: jax_privacy.keras_api.DPKerasConfig.train_steps
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.train_steps
```

````

````{py:attribute} train_size
:canonical: jax_privacy.keras_api.DPKerasConfig.train_size
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.train_size
```

````

````{py:attribute} noise_multiplier
:canonical: jax_privacy.keras_api.DPKerasConfig.noise_multiplier
:type: float | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.noise_multiplier
```

````

````{py:attribute} rescale_to_unit_norm
:canonical: jax_privacy.keras_api.DPKerasConfig.rescale_to_unit_norm
:type: bool
:value: >
   True

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.rescale_to_unit_norm
```

````

````{py:attribute} microbatch_size
:canonical: jax_privacy.keras_api.DPKerasConfig.microbatch_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.microbatch_size
```

````

````{py:attribute} seed
:canonical: jax_privacy.keras_api.DPKerasConfig.seed
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.seed
```

````

````{py:property} effective_batch_size
:canonical: jax_privacy.keras_api.DPKerasConfig.effective_batch_size
:type: int

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.effective_batch_size
```

````

````{py:method} update_with_calibrated_noise_multiplier() -> jax_privacy.keras_api.DPKerasConfig
:canonical: jax_privacy.keras_api.DPKerasConfig.update_with_calibrated_noise_multiplier

```{autodoc2-docstring} jax_privacy.keras_api.DPKerasConfig.update_with_calibrated_noise_multiplier
```

````

`````

````{py:function} make_private(model: keras.Model, params: jax_privacy.keras_api.DPKerasConfig) -> keras.Model
:canonical: jax_privacy.keras_api.make_private

```{autodoc2-docstring} jax_privacy.keras_api.make_private
```
````

````{py:data} LossFn
:canonical: jax_privacy.keras_api.LossFn
:value: >
   None

```{autodoc2-docstring} jax_privacy.keras_api.LossFn
```

````
