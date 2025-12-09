# {py:mod}`jax_privacy.matrix_factorization.sensitivity`

```{py:module} jax_privacy.matrix_factorization.sensitivity
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`single_participation_sensitivity <jax_privacy.matrix_factorization.sensitivity.single_participation_sensitivity>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.single_participation_sensitivity
    :summary:
    ```
* - {py:obj}`minsep_true_max_participations <jax_privacy.matrix_factorization.sensitivity.minsep_true_max_participations>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.minsep_true_max_participations
    :summary:
    ```
* - {py:obj}`max_participation_for_linear_fn <jax_privacy.matrix_factorization.sensitivity.max_participation_for_linear_fn>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.max_participation_for_linear_fn
    :summary:
    ```
* - {py:obj}`banded_lower_triangular_mask <jax_privacy.matrix_factorization.sensitivity.banded_lower_triangular_mask>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.banded_lower_triangular_mask
    :summary:
    ```
* - {py:obj}`banded_symmetric_mask <jax_privacy.matrix_factorization.sensitivity.banded_symmetric_mask>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.banded_symmetric_mask
    :summary:
    ```
* - {py:obj}`get_min_sep_sensitivity_upper_bound_for_X <jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound_for_X>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound_for_X
    :summary:
    ```
* - {py:obj}`get_min_sep_sensitivity_upper_bound <jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound
    :summary:
    ```
* - {py:obj}`get_sensitivity_banded_for_X <jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded_for_X>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded_for_X
    :summary:
    ```
* - {py:obj}`get_sensitivity_banded <jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded
    :summary:
    ```
* - {py:obj}`fixed_epoch_sensitivity_for_X <jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity_for_X>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity_for_X
    :summary:
    ```
* - {py:obj}`fixed_epoch_sensitivity <jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity
    :summary:
    ```
````

### API

````{py:function} single_participation_sensitivity(C: jax.numpy.ndarray) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.single_participation_sensitivity

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.single_participation_sensitivity
```
````

````{py:function} minsep_true_max_participations(n: int, min_sep: int, max_participations: typing.Optional[int] = None) -> int
:canonical: jax_privacy.matrix_factorization.sensitivity.minsep_true_max_participations

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.minsep_true_max_participations
```
````

````{py:function} max_participation_for_linear_fn(x: jax.numpy.ndarray, min_sep: int = 1, max_participations: typing.Optional[int] = None) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.max_participation_for_linear_fn

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.max_participation_for_linear_fn
```
````

````{py:function} banded_lower_triangular_mask(n: int, num_bands: int) -> jax.numpy.ndarray
:canonical: jax_privacy.matrix_factorization.sensitivity.banded_lower_triangular_mask

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.banded_lower_triangular_mask
```
````

````{py:function} banded_symmetric_mask(n: int, num_bands: int) -> jax.numpy.ndarray
:canonical: jax_privacy.matrix_factorization.sensitivity.banded_symmetric_mask

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.banded_symmetric_mask
```
````

````{py:function} get_min_sep_sensitivity_upper_bound_for_X(X: jax.numpy.ndarray, min_sep: int = 1, max_participations: typing.Optional[int] = None) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound_for_X

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound_for_X
```
````

````{py:function} get_min_sep_sensitivity_upper_bound(C: jax.numpy.ndarray, min_sep: int = 1, max_participations: typing.Optional[int] = None) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_min_sep_sensitivity_upper_bound
```
````

````{py:function} get_sensitivity_banded_for_X(X: jax.numpy.ndarray, min_sep: int = 1, max_participations: typing.Optional[int] = None) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded_for_X

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded_for_X
```
````

````{py:function} get_sensitivity_banded(C: jax.numpy.ndarray, min_sep: int = 1, max_participations: typing.Optional[int] = None) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.get_sensitivity_banded
```
````

````{py:function} fixed_epoch_sensitivity_for_X(X: jax.numpy.ndarray, epochs: int) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity_for_X

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity_for_X
```
````

````{py:function} fixed_epoch_sensitivity(C: jax.numpy.ndarray, epochs: int) -> float
:canonical: jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity

```{autodoc2-docstring} jax_privacy.matrix_factorization.sensitivity.fixed_epoch_sensitivity
```
````
