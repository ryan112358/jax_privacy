# {py:mod}`jax_privacy.matrix_factorization.checks`

```{py:module} jax_privacy.matrix_factorization.checks
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_lower_triangular <jax_privacy.matrix_factorization.checks.check_lower_triangular>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_lower_triangular
    :summary:
    ```
* - {py:obj}`check_is_matrix <jax_privacy.matrix_factorization.checks.check_is_matrix>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_is_matrix
    :summary:
    ```
* - {py:obj}`check_square <jax_privacy.matrix_factorization.checks.check_square>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_square
    :summary:
    ```
* - {py:obj}`check_finite <jax_privacy.matrix_factorization.checks.check_finite>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_finite
    :summary:
    ```
* - {py:obj}`check_symmetric <jax_privacy.matrix_factorization.checks.check_symmetric>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_symmetric
    :summary:
    ```
* - {py:obj}`check <jax_privacy.matrix_factorization.checks.check>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check
    :summary:
    ```
````

### API

````{py:function} check_lower_triangular(M: jax.numpy.ndarray, name: str = '', **allclose_kwargs)
:canonical: jax_privacy.matrix_factorization.checks.check_lower_triangular

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_lower_triangular
```
````

````{py:function} check_is_matrix(M: jax.numpy.ndarray, name: str = '')
:canonical: jax_privacy.matrix_factorization.checks.check_is_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_is_matrix
```
````

````{py:function} check_square(M: jax.numpy.ndarray, name: str = '')
:canonical: jax_privacy.matrix_factorization.checks.check_square

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_square
```
````

````{py:function} check_finite(M: jax.numpy.ndarray, name: str)
:canonical: jax_privacy.matrix_factorization.checks.check_finite

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_finite
```
````

````{py:function} check_symmetric(M: jax.numpy.ndarray, name: str, **allclose_kwargs)
:canonical: jax_privacy.matrix_factorization.checks.check_symmetric

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check_symmetric
```
````

````{py:function} check(*, A: typing.Optional[jax.numpy.ndarray] = None, B: typing.Optional[jax.numpy.ndarray] = None, C: typing.Optional[jax.numpy.ndarray] = None, X: typing.Optional[jax.numpy.ndarray] = None, **allclose_kwargs)
:canonical: jax_privacy.matrix_factorization.checks.check

```{autodoc2-docstring} jax_privacy.matrix_factorization.checks.check
```
````
