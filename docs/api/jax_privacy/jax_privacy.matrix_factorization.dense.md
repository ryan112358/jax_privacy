# {py:mod}`jax_privacy.matrix_factorization.dense`

```{py:module} jax_privacy.matrix_factorization.dense
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`per_query_error <jax_privacy.matrix_factorization.dense.per_query_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.per_query_error
    :summary:
    ```
* - {py:obj}`max_error <jax_privacy.matrix_factorization.dense.max_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.max_error
    :summary:
    ```
* - {py:obj}`mean_error <jax_privacy.matrix_factorization.dense.mean_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.mean_error
    :summary:
    ```
* - {py:obj}`get_orthogonal_mask <jax_privacy.matrix_factorization.dense.get_orthogonal_mask>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.get_orthogonal_mask
    :summary:
    ```
* - {py:obj}`strategy_from_X <jax_privacy.matrix_factorization.dense.strategy_from_X>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.strategy_from_X
    :summary:
    ```
* - {py:obj}`pg_tol_termination_fn <jax_privacy.matrix_factorization.dense.pg_tol_termination_fn>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.pg_tol_termination_fn
    :summary:
    ```
* - {py:obj}`optimize <jax_privacy.matrix_factorization.dense.optimize>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.optimize
    :summary:
    ```
````

### API

````{py:function} per_query_error(*, strategy_matrix: jax.Array | None = None, noising_matrix: jax.Array | None = None, workload_matrix: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.per_query_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.per_query_error
```
````

````{py:function} max_error(*, strategy_matrix: jax.Array | None = None, noising_matrix: jax.Array | None = None, workload_matrix: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.max_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.max_error
```
````

````{py:function} mean_error(*, strategy_matrix: jax.Array | None = None, noising_matrix: jax.Array | None = None, workload_matrix: jax.Array | None = None, skip_checks: bool = False) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.mean_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.mean_error
```
````

````{py:function} get_orthogonal_mask(n: int, epochs: int = 1) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.get_orthogonal_mask

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.get_orthogonal_mask
```
````

````{py:function} strategy_from_X(X: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.strategy_from_X

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.strategy_from_X
```
````

````{py:function} pg_tol_termination_fn(step_info: jax_privacy.matrix_factorization.optimization.CallbackArgs) -> bool
:canonical: jax_privacy.matrix_factorization.dense.pg_tol_termination_fn

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.pg_tol_termination_fn
```
````

````{py:function} optimize(n: int, *, epochs: int = 1, bands: int | None = None, equal_norm: bool = False, A: jax.Array | None = None, max_optimizer_steps: int = 10000, callback: jax_privacy.matrix_factorization.optimization.CallbackFnType = pg_tol_termination_fn) -> jax.Array
:canonical: jax_privacy.matrix_factorization.dense.optimize

```{autodoc2-docstring} jax_privacy.matrix_factorization.dense.optimize
```
````
