# {py:mod}`jax_privacy.matrix_factorization.toeplitz`

```{py:module} jax_privacy.matrix_factorization.toeplitz
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ErrorOrLossFn <jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pad_coefs_to_n <jax_privacy.matrix_factorization.toeplitz.pad_coefs_to_n>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.pad_coefs_to_n
    :summary:
    ```
* - {py:obj}`inverse_as_streaming_matrix <jax_privacy.matrix_factorization.toeplitz.inverse_as_streaming_matrix>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.inverse_as_streaming_matrix
    :summary:
    ```
* - {py:obj}`optimal_max_error_strategy_coefs <jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs
    :summary:
    ```
* - {py:obj}`optimal_max_error_noising_coefs <jax_privacy.matrix_factorization.toeplitz.optimal_max_error_noising_coefs>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimal_max_error_noising_coefs
    :summary:
    ```
* - {py:obj}`materialize_lower_triangular <jax_privacy.matrix_factorization.toeplitz.materialize_lower_triangular>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.materialize_lower_triangular
    :summary:
    ```
* - {py:obj}`solve_banded <jax_privacy.matrix_factorization.toeplitz.solve_banded>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.solve_banded
    :summary:
    ```
* - {py:obj}`multiply <jax_privacy.matrix_factorization.toeplitz.multiply>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.multiply
    :summary:
    ```
* - {py:obj}`inverse_coef <jax_privacy.matrix_factorization.toeplitz.inverse_coef>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.inverse_coef
    :summary:
    ```
* - {py:obj}`sensitivity_squared <jax_privacy.matrix_factorization.toeplitz.sensitivity_squared>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.sensitivity_squared
    :summary:
    ```
* - {py:obj}`minsep_sensitivity_squared <jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared
    :summary:
    ```
* - {py:obj}`per_query_error <jax_privacy.matrix_factorization.toeplitz.per_query_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.per_query_error
    :summary:
    ```
* - {py:obj}`max_error <jax_privacy.matrix_factorization.toeplitz.max_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.max_error
    :summary:
    ```
* - {py:obj}`mean_error <jax_privacy.matrix_factorization.toeplitz.mean_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.mean_error
    :summary:
    ```
* - {py:obj}`loss <jax_privacy.matrix_factorization.toeplitz.loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.loss
    :summary:
    ```
* - {py:obj}`optimize_banded_toeplitz <jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz
    :summary:
    ```
* - {py:obj}`optimize_coefs_for_amplifications <jax_privacy.matrix_factorization.toeplitz.optimize_coefs_for_amplifications>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimize_coefs_for_amplifications
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`mean_loss <jax_privacy.matrix_factorization.toeplitz.mean_loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.mean_loss
    :summary:
    ```
* - {py:obj}`max_loss <jax_privacy.matrix_factorization.toeplitz.max_loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.max_loss
    :summary:
    ```
````

### API

````{py:function} pad_coefs_to_n(coef: jax.Array, n: int | None = None) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.pad_coefs_to_n

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.pad_coefs_to_n
```
````

````{py:function} inverse_as_streaming_matrix(coef: jax.Array, column_normalize_for_n: int | None = None) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.toeplitz.inverse_as_streaming_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.inverse_as_streaming_matrix
```
````

````{py:function} optimal_max_error_strategy_coefs(n: int) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimal_max_error_strategy_coefs
```
````

````{py:function} optimal_max_error_noising_coefs(n: int) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.optimal_max_error_noising_coefs

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimal_max_error_noising_coefs
```
````

````{py:function} materialize_lower_triangular(coef: jax.Array, n: int | None = None) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.materialize_lower_triangular

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.materialize_lower_triangular
```
````

````{py:function} solve_banded(coef: jax.Array, rhs: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.solve_banded

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.solve_banded
```
````

````{py:function} multiply(lhs_coef: jax.Array, rhs_coef: jax.Array, n: int | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.multiply

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.multiply
```
````

````{py:function} inverse_coef(coef: jax.Array, n: int | None = None) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.inverse_coef

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.inverse_coef
```
````

````{py:function} sensitivity_squared(coef: jax.Array, n: int | None = None) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.sensitivity_squared

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.sensitivity_squared
```
````

````{py:function} minsep_sensitivity_squared(strategy_coef: jax.Array, min_sep: int, max_participations: int | None = None, n: int | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.minsep_sensitivity_squared
```
````

````{py:function} per_query_error(*, strategy_coef: jax.Array | None = None, noising_coef: jax.Array | None = None, n: int | None = None, workload_coef: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.per_query_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.per_query_error
```
````

````{py:function} max_error(*, strategy_coef: jax.Array | None = None, noising_coef: jax.Array | None = None, n: int | None = None, workload_coef: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.max_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.max_error
```
````

````{py:function} mean_error(*, strategy_coef: jax.Array | None = None, noising_coef: jax.Array | None = None, n: int | None = None, workload_coef: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.mean_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.mean_error
```
````

````{py:class} ErrorOrLossFn
:canonical: jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn
```

````

````{py:function} loss(strategy_coef: jax.Array, n: int | None = None, error_fn: jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn = mean_error) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.loss
```
````

````{py:data} mean_loss
:canonical: jax_privacy.matrix_factorization.toeplitz.mean_loss
:value: >
   'partial(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.mean_loss
```

````

````{py:data} max_loss
:canonical: jax_privacy.matrix_factorization.toeplitz.max_loss
:value: >
   'partial(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.max_loss
```

````

````{py:function} optimize_banded_toeplitz(n: int, bands: int, strategy_coef: jax.Array | None = None, max_optimizer_steps: int = 250, loss_fn: jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn = mean_loss) -> jax.Array
:canonical: jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimize_banded_toeplitz
```
````

````{py:function} optimize_coefs_for_amplifications(n: int, *, dataset_size: int, expected_batch_size: int, epsilon: float, delta: float, max_optimizer_steps: int = 250, loss_fn: jax_privacy.matrix_factorization.toeplitz.ErrorOrLossFn = mean_loss) -> tuple[jax.Array, float]
:canonical: jax_privacy.matrix_factorization.toeplitz.optimize_coefs_for_amplifications

```{autodoc2-docstring} jax_privacy.matrix_factorization.toeplitz.optimize_coefs_for_amplifications
```
````
