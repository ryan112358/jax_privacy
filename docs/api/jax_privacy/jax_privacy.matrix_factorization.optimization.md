# {py:mod}`jax_privacy.matrix_factorization.optimization`

```{py:module} jax_privacy.matrix_factorization.optimization
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CallbackArgs <jax_privacy.matrix_factorization.optimization.CallbackArgs>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`jax_enable_x64 <jax_privacy.matrix_factorization.optimization.jax_enable_x64>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.jax_enable_x64
    :summary:
    ```
* - {py:obj}`optimize <jax_privacy.matrix_factorization.optimization.optimize>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.optimize
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ParamT <jax_privacy.matrix_factorization.optimization.ParamT>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.ParamT
    :summary:
    ```
* - {py:obj}`DEFAULT_OPTIMIZER <jax_privacy.matrix_factorization.optimization.DEFAULT_OPTIMIZER>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.DEFAULT_OPTIMIZER
    :summary:
    ```
* - {py:obj}`CallbackFnType <jax_privacy.matrix_factorization.optimization.CallbackFnType>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackFnType
    :summary:
    ```
````

### API

````{py:data} ParamT
:canonical: jax_privacy.matrix_factorization.optimization.ParamT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.ParamT
```

````

````{py:data} DEFAULT_OPTIMIZER
:canonical: jax_privacy.matrix_factorization.optimization.DEFAULT_OPTIMIZER
:value: >
   'lbfgs(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.DEFAULT_OPTIMIZER
```

````

`````{py:class} CallbackArgs
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs
```

````{py:attribute} step
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs.step
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs.step
```

````

````{py:attribute} loss
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs.loss
:type: jax.numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs.loss
```

````

````{py:attribute} grad
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs.grad
:type: chex.ArrayTree | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs.grad
```

````

````{py:attribute} params
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs.params
:type: chex.ArrayTree
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs.params
```

````

````{py:attribute} state
:canonical: jax_privacy.matrix_factorization.optimization.CallbackArgs.state
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackArgs.state
```

````

`````

````{py:data} CallbackFnType
:canonical: jax_privacy.matrix_factorization.optimization.CallbackFnType
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.CallbackFnType
```

````

````{py:function} jax_enable_x64(fn: collections.abc.Callable[..., typing.Any]) -> collections.abc.Callable[..., typing.Any]
:canonical: jax_privacy.matrix_factorization.optimization.jax_enable_x64

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.jax_enable_x64
```
````

````{py:function} optimize(loss_fn: collections.abc.Callable[[jax_privacy.matrix_factorization.optimization.ParamT], jax.numpy.ndarray | tuple[jax.numpy.ndarray, jax_privacy.matrix_factorization.optimization.ParamT]], params: jax_privacy.matrix_factorization.optimization.ParamT, *, max_optimizer_steps: int = 250, grad: bool = False, callback: jax_privacy.matrix_factorization.optimization.CallbackFnType = lambda _: None, optimizer: optax.GradientTransformationExtraArgs = DEFAULT_OPTIMIZER) -> jax_privacy.matrix_factorization.optimization.ParamT
:canonical: jax_privacy.matrix_factorization.optimization.optimize

```{autodoc2-docstring} jax_privacy.matrix_factorization.optimization.optimize
```
````
