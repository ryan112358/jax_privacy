# {py:mod}`jax_privacy.matrix_factorization.banded`

```{py:module} jax_privacy.matrix_factorization.banded
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ColumnNormalizedBanded <jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`minsep_sensitivity_squared <jax_privacy.matrix_factorization.banded.minsep_sensitivity_squared>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.minsep_sensitivity_squared
    :summary:
    ```
* - {py:obj}`per_query_error <jax_privacy.matrix_factorization.banded.per_query_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.per_query_error
    :summary:
    ```
* - {py:obj}`optimize <jax_privacy.matrix_factorization.banded.optimize>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.optimize
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mean_error <jax_privacy.matrix_factorization.banded.mean_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.mean_error
    :summary:
    ```
* - {py:obj}`last_error <jax_privacy.matrix_factorization.banded.last_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.last_error
    :summary:
    ```
* - {py:obj}`max_error <jax_privacy.matrix_factorization.banded.max_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.max_error
    :summary:
    ```
````

### API

`````{py:class} ColumnNormalizedBanded
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded
```

````{py:attribute} params
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.params
:type: jax.numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.params
```

````

````{py:property} n
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.n
:type: int

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.n
```

````

````{py:property} bands
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.bands
:type: int

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.bands
```

````

````{py:method} from_banded_toeplitz(n: int, coefs: jax.numpy.ndarray) -> jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.from_banded_toeplitz
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.from_banded_toeplitz
```

````

````{py:method} default(n: int, bands: int) -> jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.default
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.default
```

````

````{py:method} materialize() -> jax.numpy.ndarray
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.materialize

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.materialize
```

````

````{py:method} inverse_as_streaming_matrix() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.inverse_as_streaming_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded.inverse_as_streaming_matrix
```

````

`````

````{py:function} minsep_sensitivity_squared(strategy: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded, min_sep: int, max_participations: int | None = None, n: int | None = None, skip_checks: bool = False) -> int
:canonical: jax_privacy.matrix_factorization.banded.minsep_sensitivity_squared

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.minsep_sensitivity_squared
```
````

````{py:function} per_query_error(C: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded, A: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix | None = None, scan_fn: typing.Any = jax.lax.scan) -> jax.numpy.ndarray
:canonical: jax_privacy.matrix_factorization.banded.per_query_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.per_query_error
```
````

````{py:data} mean_error
:canonical: jax_privacy.matrix_factorization.banded.mean_error
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.mean_error
```

````

````{py:data} last_error
:canonical: jax_privacy.matrix_factorization.banded.last_error
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.last_error
```

````

````{py:data} max_error
:canonical: jax_privacy.matrix_factorization.banded.max_error
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.max_error
```

````

````{py:function} optimize(n: int, *, bands: int, C: jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded | None = None, A: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix | None = None, max_optimizer_steps: int = 100, reduction_fn: collections.abc.Callable[[jax.numpy.ndarray], jax.numpy.ndarray] = jnp.mean, scan_fn: typing.Any = jax.lax.scan, callback: jax_privacy.matrix_factorization.optimization.CallbackFnType = lambda _: None) -> jax_privacy.matrix_factorization.banded.ColumnNormalizedBanded
:canonical: jax_privacy.matrix_factorization.banded.optimize

```{autodoc2-docstring} jax_privacy.matrix_factorization.banded.optimize
```
````
