# {py:mod}`jax_privacy.matrix_factorization.streaming_matrix`

```{py:module} jax_privacy.matrix_factorization.streaming_matrix
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StreamingMatrix <jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`scale_rows_and_columns <jax_privacy.matrix_factorization.streaming_matrix.scale_rows_and_columns>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.scale_rows_and_columns
    :summary:
    ```
* - {py:obj}`multiply_array <jax_privacy.matrix_factorization.streaming_matrix.multiply_array>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.multiply_array
    :summary:
    ```
* - {py:obj}`multiply_streaming_matrices <jax_privacy.matrix_factorization.streaming_matrix.multiply_streaming_matrices>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.multiply_streaming_matrices
    :summary:
    ```
* - {py:obj}`identity <jax_privacy.matrix_factorization.streaming_matrix.identity>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.identity
    :summary:
    ```
* - {py:obj}`prefix_sum <jax_privacy.matrix_factorization.streaming_matrix.prefix_sum>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.prefix_sum
    :summary:
    ```
* - {py:obj}`diagonal <jax_privacy.matrix_factorization.streaming_matrix.diagonal>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.diagonal
    :summary:
    ```
* - {py:obj}`momentum_sgd_matrix <jax_privacy.matrix_factorization.streaming_matrix.momentum_sgd_matrix>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.momentum_sgd_matrix
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`State <jax_privacy.matrix_factorization.streaming_matrix.State>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.State
    :summary:
    ```
* - {py:obj}`Shape <jax_privacy.matrix_factorization.streaming_matrix.Shape>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.Shape
    :summary:
    ```
* - {py:obj}`ShapePyTree <jax_privacy.matrix_factorization.streaming_matrix.ShapePyTree>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.ShapePyTree
    :summary:
    ```
* - {py:obj}`T <jax_privacy.matrix_factorization.streaming_matrix.T>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.T
    :summary:
    ```
````

### API

````{py:data} State
:canonical: jax_privacy.matrix_factorization.streaming_matrix.State
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.State
```

````

````{py:data} Shape
:canonical: jax_privacy.matrix_factorization.streaming_matrix.Shape
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.Shape
```

````

````{py:data} ShapePyTree
:canonical: jax_privacy.matrix_factorization.streaming_matrix.ShapePyTree
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.ShapePyTree
```

````

`````{py:class} StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
```

````{py:attribute} init_multiply
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.init_multiply
:type: collections.abc.Callable[[chex.ArrayTree], jax_privacy.matrix_factorization.streaming_matrix.State]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.init_multiply
```

````

````{py:attribute} multiply_next
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.multiply_next
:type: collections.abc.Callable[[chex.ArrayTree, jax_privacy.matrix_factorization.streaming_matrix.State], tuple[chex.ArrayTree, jax_privacy.matrix_factorization.streaming_matrix.State]]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.multiply_next
```

````

````{py:method} from_array_implementation(init_multiply_fn: collections.abc.Callable[[jax.Array | jax.ShapeDtypeStruct], jax_privacy.matrix_factorization.streaming_matrix.State], multiply_next_fn: collections.abc.Callable[[jax.Array, jax_privacy.matrix_factorization.streaming_matrix.State], tuple[jax.Array, jax_privacy.matrix_factorization.streaming_matrix.State]]) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.from_array_implementation
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.from_array_implementation
```

````

````{py:method} materialize(n: int) -> jax.Array
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.materialize

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.materialize
```

````

````{py:method} row_norms_squared(n: int, scan_fn=jax.lax.scan) -> jax.Array
:canonical: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.row_norms_squared

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix.row_norms_squared
```

````

`````

````{py:function} scale_rows_and_columns(matrix: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix, row_scale: jax.Array | None = None, col_scale: jax.Array | None = None) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.scale_rows_and_columns

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.scale_rows_and_columns
```
````

````{py:function} multiply_array(A: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix, x: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.streaming_matrix.multiply_array

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.multiply_array
```
````

````{py:function} multiply_streaming_matrices(A: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix, B: jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.multiply_streaming_matrices

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.multiply_streaming_matrices
```
````

````{py:function} identity() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.identity

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.identity
```
````

````{py:function} prefix_sum() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.prefix_sum

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.prefix_sum
```
````

````{py:function} diagonal(diag: jax.Array) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.diagonal

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.diagonal
```
````

````{py:function} momentum_sgd_matrix(momentum: float = 0, learning_rates: jax.Array | None = None) -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.streaming_matrix.momentum_sgd_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.momentum_sgd_matrix
```
````

````{py:data} T
:canonical: jax_privacy.matrix_factorization.streaming_matrix.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.streaming_matrix.T
```

````
