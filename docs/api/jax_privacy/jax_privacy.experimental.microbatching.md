# {py:mod}`jax_privacy.experimental.microbatching`

```{py:module} jax_privacy.experimental.microbatching
```

```{autodoc2-docstring} jax_privacy.experimental.microbatching
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AccumulationType <jax_privacy.experimental.microbatching.AccumulationType>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`microbatch <jax_privacy.experimental.microbatching.microbatch>`
  - ```{autodoc2-docstring} jax_privacy.experimental.microbatching.microbatch
    :summary:
    ```
* - {py:obj}`compute_early_stopping_order <jax_privacy.experimental.microbatching.compute_early_stopping_order>`
  - ```{autodoc2-docstring} jax_privacy.experimental.microbatching.compute_early_stopping_order
    :summary:
    ```
* - {py:obj}`verify_early_stopping_order <jax_privacy.experimental.microbatching.verify_early_stopping_order>`
  - ```{autodoc2-docstring} jax_privacy.experimental.microbatching.verify_early_stopping_order
    :summary:
    ```
````

### API

`````{py:class} AccumulationType(*args, **kwds)
:canonical: jax_privacy.experimental.microbatching.AccumulationType

Bases: {py:obj}`enum.Enum`

````{py:attribute} SUM
:canonical: jax_privacy.experimental.microbatching.AccumulationType.SUM
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.experimental.microbatching.AccumulationType.SUM
```

````

````{py:attribute} MEAN
:canonical: jax_privacy.experimental.microbatching.AccumulationType.MEAN
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.experimental.microbatching.AccumulationType.MEAN
```

````

````{py:attribute} CONCAT
:canonical: jax_privacy.experimental.microbatching.AccumulationType.CONCAT
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.experimental.microbatching.AccumulationType.CONCAT
```

````

`````

````{py:function} microbatch(fun: typing.Callable, batch_argnums: int | typing.Sequence[int], microbatch_size: int | None, accumulation_type: typing.Any = AccumulationType.SUM, dtype: jax.typing.DTypeLike | None = None) -> typing.Callable
:canonical: jax_privacy.experimental.microbatching.microbatch

```{autodoc2-docstring} jax_privacy.experimental.microbatching.microbatch
```
````

````{py:function} compute_early_stopping_order(batch_size: int, microbatch_size: int | None) -> numpy.ndarray
:canonical: jax_privacy.experimental.microbatching.compute_early_stopping_order

```{autodoc2-docstring} jax_privacy.experimental.microbatching.compute_early_stopping_order
```
````

````{py:function} verify_early_stopping_order(is_padding_example: jax.Array, microbatch_size: int | None) -> bool
:canonical: jax_privacy.experimental.microbatching.verify_early_stopping_order

```{autodoc2-docstring} jax_privacy.experimental.microbatching.verify_early_stopping_order
```
````
