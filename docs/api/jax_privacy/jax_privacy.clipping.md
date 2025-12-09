# {py:mod}`jax_privacy.clipping`

```{py:module} jax_privacy.clipping
```

```{autodoc2-docstring} jax_privacy.clipping
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BoundedSensitivityCallable <jax_privacy.clipping.BoundedSensitivityCallable>`
  - ```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clip_pytree <jax_privacy.clipping.clip_pytree>`
  - ```{autodoc2-docstring} jax_privacy.clipping.clip_pytree
    :summary:
    ```
* - {py:obj}`clipped_fun <jax_privacy.clipping.clipped_fun>`
  - ```{autodoc2-docstring} jax_privacy.clipping.clipped_fun
    :summary:
    ```
* - {py:obj}`clipped_grad <jax_privacy.clipping.clipped_grad>`
  - ```{autodoc2-docstring} jax_privacy.clipping.clipped_grad
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PyTree <jax_privacy.clipping.PyTree>`
  - ```{autodoc2-docstring} jax_privacy.clipping.PyTree
    :summary:
    ```
* - {py:obj}`AuxiliaryOutput <jax_privacy.clipping.AuxiliaryOutput>`
  - ```{autodoc2-docstring} jax_privacy.clipping.AuxiliaryOutput
    :summary:
    ```
````

### API

````{py:data} PyTree
:canonical: jax_privacy.clipping.PyTree
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.clipping.PyTree
```

````

````{py:data} AuxiliaryOutput
:canonical: jax_privacy.clipping.AuxiliaryOutput
:value: >
   'namedtuple(...)'

```{autodoc2-docstring} jax_privacy.clipping.AuxiliaryOutput
```

````

`````{py:class} BoundedSensitivityCallable
:canonical: jax_privacy.clipping.BoundedSensitivityCallable

```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable
```

````{py:attribute} fun
:canonical: jax_privacy.clipping.BoundedSensitivityCallable.fun
:type: typing.Callable[..., typing.Any]
:value: >
   None

```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable.fun
```

````

````{py:attribute} l2_norm_bound
:canonical: jax_privacy.clipping.BoundedSensitivityCallable.l2_norm_bound
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable.l2_norm_bound
```

````

````{py:attribute} has_aux
:canonical: jax_privacy.clipping.BoundedSensitivityCallable.has_aux
:type: bool
:value: >
   None

```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable.has_aux
```

````

````{py:method} sensitivity(neighboring_relation: dp_accounting.NeighboringRelation = _REPLACE_SPECIAL)
:canonical: jax_privacy.clipping.BoundedSensitivityCallable.sensitivity

```{autodoc2-docstring} jax_privacy.clipping.BoundedSensitivityCallable.sensitivity
```

````

`````

````{py:function} clip_pytree(pytree: jax_privacy.clipping.PyTree, clip_norm: float, rescale_to_unit_norm: bool = False, nan_safe: bool = True, return_zero: bool = False)
:canonical: jax_privacy.clipping.clip_pytree

```{autodoc2-docstring} jax_privacy.clipping.clip_pytree
```
````

````{py:function} clipped_fun(fun: typing.Callable, has_aux: bool = False, *, batch_argnums: int | collections.abc.Sequence[int] = 0, keep_batch_dim: bool = True, l2_clip_norm: float = 1.0, rescale_to_unit_norm: bool = False, normalize_by: float = 1.0, return_norms: bool = False, microbatch_size: int | None = None, nan_safe: bool = True, dtype: jax.typing.DTypeLike | None = None, prng_argnum: int | None = None, spmd_axis_name: str | None = None) -> jax_privacy.clipping.BoundedSensitivityCallable
:canonical: jax_privacy.clipping.clipped_fun

```{autodoc2-docstring} jax_privacy.clipping.clipped_fun
```
````

````{py:function} clipped_grad(fun: typing.Callable, argnums: int | collections.abc.Sequence[int] = 0, has_aux: bool = False, *, l2_clip_norm: float, rescale_to_unit_norm: bool = False, normalize_by: float = 1.0, batch_argnums: int | collections.abc.Sequence[int] = 1, keep_batch_dim: bool = True, return_values: bool = False, return_grad_norms: bool = False, pre_clipping_transform: typing.Callable[[jax_privacy.clipping.PyTree], jax_privacy.clipping.PyTree] = lambda x: x, microbatch_size: int | None = None, nan_safe: bool = True, dtype: jax.typing.DTypeLike | None = None, prng_argnum: int | None = None, spmd_axis_name: str | None = None) -> jax_privacy.clipping.BoundedSensitivityCallable
:canonical: jax_privacy.clipping.clipped_grad

```{autodoc2-docstring} jax_privacy.clipping.clipped_grad
```
````
