# {py:mod}`jax_privacy.noise_addition`

```{py:module} jax_privacy.noise_addition
```

```{autodoc2-docstring} jax_privacy.noise_addition
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SupportedStrategies <jax_privacy.noise_addition.SupportedStrategies>`
  - ```{autodoc2-docstring} jax_privacy.noise_addition.SupportedStrategies
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`matrix_factorization_privatizer <jax_privacy.noise_addition.matrix_factorization_privatizer>`
  - ```{autodoc2-docstring} jax_privacy.noise_addition.matrix_factorization_privatizer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`gaussian_privatizer <jax_privacy.noise_addition.gaussian_privatizer>`
  - ```{autodoc2-docstring} jax_privacy.noise_addition.gaussian_privatizer
    :summary:
    ```
````

### API

`````{py:class} SupportedStrategies(*args, **kwds)
:canonical: jax_privacy.noise_addition.SupportedStrategies

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} jax_privacy.noise_addition.SupportedStrategies
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.noise_addition.SupportedStrategies.__init__
```

````{py:attribute} DEFAULT
:canonical: jax_privacy.noise_addition.SupportedStrategies.DEFAULT
:value: >
   '_IntermediateStrategy(...)'

```{autodoc2-docstring} jax_privacy.noise_addition.SupportedStrategies.DEFAULT
```

````

````{py:attribute} ZERO
:canonical: jax_privacy.noise_addition.SupportedStrategies.ZERO
:value: >
   '_IntermediateStrategy(...)'

```{autodoc2-docstring} jax_privacy.noise_addition.SupportedStrategies.ZERO
```

````

`````

````{py:function} matrix_factorization_privatizer(noising_matrix: jax.typing.ArrayLike | jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix, *, stddev: float, prng_key: jax.Array | int | None = None, intermediate_strategy: jax_privacy.noise_addition.SupportedStrategies = SupportedStrategies.DEFAULT) -> optax.GradientTransformation
:canonical: jax_privacy.noise_addition.matrix_factorization_privatizer

```{autodoc2-docstring} jax_privacy.noise_addition.matrix_factorization_privatizer
```
````

````{py:data} gaussian_privatizer
:canonical: jax_privacy.noise_addition.gaussian_privatizer
:value: >
   'partial(...)'

```{autodoc2-docstring} jax_privacy.noise_addition.gaussian_privatizer
```

````
