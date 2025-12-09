# {py:mod}`jax_privacy.matrix_factorization.buffered_toeplitz`

```{py:module} jax_privacy.matrix_factorization.buffered_toeplitz
```

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StreamingMatrixBuilder <jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder
    :summary:
    ```
* - {py:obj}`BufferedToeplitz <jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
    :summary:
    ```
* - {py:obj}`LossFn <jax_privacy.matrix_factorization.buffered_toeplitz.LossFn>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn
    :summary:
    ```
* - {py:obj}`Parameterization <jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`check_float64_dtype <jax_privacy.matrix_factorization.buffered_toeplitz.check_float64_dtype>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.check_float64_dtype
    :summary:
    ```
* - {py:obj}`min_buf_decay_gap <jax_privacy.matrix_factorization.buffered_toeplitz.min_buf_decay_gap>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.min_buf_decay_gap
    :summary:
    ```
* - {py:obj}`get_init_blt <jax_privacy.matrix_factorization.buffered_toeplitz.get_init_blt>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.get_init_blt
    :summary:
    ```
* - {py:obj}`optimize_loss <jax_privacy.matrix_factorization.buffered_toeplitz.optimize_loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.optimize_loss
    :summary:
    ```
* - {py:obj}`optimize <jax_privacy.matrix_factorization.buffered_toeplitz.optimize>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.optimize
    :summary:
    ```
* - {py:obj}`geometric_sum <jax_privacy.matrix_factorization.buffered_toeplitz.geometric_sum>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.geometric_sum
    :summary:
    ```
* - {py:obj}`require_buf_decay_less_eq_one <jax_privacy.matrix_factorization.buffered_toeplitz.require_buf_decay_less_eq_one>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.require_buf_decay_less_eq_one
    :summary:
    ```
* - {py:obj}`require_output_scale_gt_zero <jax_privacy.matrix_factorization.buffered_toeplitz.require_output_scale_gt_zero>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.require_output_scale_gt_zero
    :summary:
    ```
* - {py:obj}`blt_pair_from_theta_pair <jax_privacy.matrix_factorization.buffered_toeplitz.blt_pair_from_theta_pair>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.blt_pair_from_theta_pair
    :summary:
    ```
* - {py:obj}`sensitivity_squared <jax_privacy.matrix_factorization.buffered_toeplitz.sensitivity_squared>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.sensitivity_squared
    :summary:
    ```
* - {py:obj}`robust_max_error_Gamma_j <jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_j>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_j
    :summary:
    ```
* - {py:obj}`robust_max_error_Gamma_jk <jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_jk>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_jk
    :summary:
    ```
* - {py:obj}`iteration_error <jax_privacy.matrix_factorization.buffered_toeplitz.iteration_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.iteration_error
    :summary:
    ```
* - {py:obj}`max_error <jax_privacy.matrix_factorization.buffered_toeplitz.max_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.max_error
    :summary:
    ```
* - {py:obj}`limit_max_error <jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_error>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_error
    :summary:
    ```
* - {py:obj}`max_loss <jax_privacy.matrix_factorization.buffered_toeplitz.max_loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.max_loss
    :summary:
    ```
* - {py:obj}`limit_max_loss <jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_loss>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_loss
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThetaPairType <jax_privacy.matrix_factorization.buffered_toeplitz.ThetaPairType>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.ThetaPairType
    :summary:
    ```
* - {py:obj}`ScalarFloat <jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat>`
  - ```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
    :summary:
    ```
````

### API

````{py:data} ThetaPairType
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.ThetaPairType
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.ThetaPairType
```

````

````{py:data} ScalarFloat
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
```

````

````{py:function} check_float64_dtype(blt: BufferedToeplitz)
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.check_float64_dtype

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.check_float64_dtype
```
````

`````{py:class} StreamingMatrixBuilder
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder
```

````{py:attribute} buf_decay
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.buf_decay
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.buf_decay
```

````

````{py:attribute} output_scale
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.output_scale
:type: numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.output_scale
```

````

````{py:property} dtype
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.dtype
:type: numpy.dtype

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.dtype
```

````

````{py:method} build() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.build

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.build
```

````

````{py:method} build_inverse() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.build_inverse

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.StreamingMatrixBuilder.build_inverse
```

````

`````

`````{py:class} BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
```

````{py:attribute} buf_decay
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.buf_decay
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.buf_decay
```

````

````{py:attribute} output_scale
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.output_scale
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.output_scale
```

````

````{py:method} validate()
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.validate

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.validate
```

````

````{py:method} build(buf_decay: typing.Any, output_scale: typing.Any, dtype: jax.typing.DTypeLike = jnp.float64) -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.build
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.build
```

````

````{py:method} from_rational_approx_to_sqrt_x(num_buffers: int, *, max_buf_decay: float = 1.0, max_pillutla_score: float | None = None, buf_decay_scale: float = 1.6, buf_decay_shift: int = -1) -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.from_rational_approx_to_sqrt_x
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.from_rational_approx_to_sqrt_x
```

````

````{py:method} canonicalize() -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.canonicalize

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.canonicalize
```

````

````{py:property} dtype
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.dtype
:type: jax.typing.DTypeLike

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.dtype
```

````

````{py:method} toeplitz_coefs(n: int) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.toeplitz_coefs

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.toeplitz_coefs
```

````

````{py:method} materialize(n: int) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.materialize

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.materialize
```

````

````{py:method} inverse(skip_checks: bool = False) -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.inverse

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.inverse
```

````

````{py:method} pillutla_score() -> jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.pillutla_score

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.pillutla_score
```

````

````{py:method} as_streaming_matrix() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.as_streaming_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.as_streaming_matrix
```

````

````{py:method} inverse_as_streaming_matrix() -> jax_privacy.matrix_factorization.streaming_matrix.StreamingMatrix
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.inverse_as_streaming_matrix

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz.inverse_as_streaming_matrix
```

````

`````

````{py:function} min_buf_decay_gap(buf_decay: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.min_buf_decay_gap

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.min_buf_decay_gap
```
````

`````{py:class} LossFn
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn
```

````{py:attribute} error_for_inv
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.error_for_inv
:type: collections.abc.Callable[[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz], jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.error_for_inv
```

````

````{py:attribute} sensitivity_squared
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.sensitivity_squared
:type: collections.abc.Callable[[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz], jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.sensitivity_squared
```

````

````{py:attribute} n
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.n
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.n
```

````

````{py:attribute} min_sep
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.min_sep
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.min_sep
```

````

````{py:attribute} max_participations
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.max_participations
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.max_participations
```

````

````{py:attribute} penalty_strength
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalty_strength
:type: float
:value: >
   1e-08

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalty_strength
```

````

````{py:attribute} penalty_multipliers
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalty_multipliers
:type: dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalty_multipliers
```

````

````{py:attribute} max_second_coef
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.max_second_coef
:type: float
:value: >
   1.0

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.max_second_coef
```

````

````{py:attribute} min_theta_gap
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.min_theta_gap
:type: float
:value: >
   1e-12

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.min_theta_gap
```

````

````{py:method} build_closed_form_single_participation(n: int, **kwargs) -> jax_privacy.matrix_factorization.buffered_toeplitz.LossFn
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.build_closed_form_single_participation
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.build_closed_form_single_participation
```

````

````{py:method} build_min_sep(n: int, error: str = 'max', min_sep: int = 1, max_participations: int | None = None, **kwargs) -> jax_privacy.matrix_factorization.buffered_toeplitz.LossFn
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.build_min_sep
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.build_min_sep
```

````

````{py:method} compute_penalties(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, inv_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz) -> dict[str, jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat]
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.compute_penalties

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.compute_penalties
```

````

````{py:method} penalized_loss(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, inv_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, normalize_by_approx_optimal_loss: bool = True) -> jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalized_loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.penalized_loss
```

````

````{py:method} loss(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, skip_checks: bool = False) -> jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.LossFn.loss
```

````

`````

`````{py:class} Parameterization
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization
```

````{py:attribute} params_from_blt
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.params_from_blt
:type: collections.abc.Callable[[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz], chex.ArrayTree]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.params_from_blt
```

````

````{py:attribute} blt_and_inverse_from_params
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.blt_and_inverse_from_params
:type: collections.abc.Callable[[typing.Any], tuple[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz]]
:value: >
   None

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.blt_and_inverse_from_params
```

````

````{py:method} strategy_blt() -> jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.strategy_blt
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.strategy_blt
```

````

````{py:method} buf_decay_pair() -> jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.buf_decay_pair
:classmethod:

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.buf_decay_pair
```

````

````{py:method} get_loss_fn(loss_fn: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn) -> collections.abc.Callable[[chex.ArrayTree], jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat]
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.get_loss_fn

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization.get_loss_fn
```

````

`````

````{py:function} get_init_blt(num_buffers: int = 3, init_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz | None = None) -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.get_init_blt

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.get_init_blt
```
````

````{py:function} optimize_loss(loss_fn: jax_privacy.matrix_factorization.buffered_toeplitz.LossFn, num_buffers: int = 1, init_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz | None = None, parameterization: jax_privacy.matrix_factorization.buffered_toeplitz.Parameterization | None = None, **kwargs) -> tuple[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, jax_privacy.matrix_factorization.buffered_toeplitz.ScalarFloat]
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.optimize_loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.optimize_loss
```
````

````{py:function} optimize(*, n: int, min_sep: int = 1, max_participations: int | None = 1, error: str = 'max', min_buffers: int = 0, max_buffers: int = 10, rtol: float = 1.01, **kwargs) -> jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.optimize

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.optimize
```
````

````{py:function} geometric_sum(a: jax.Array, r: jax.Array, num: chex.Numeric = jnp.inf) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.geometric_sum

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.geometric_sum
```
````

````{py:function} require_buf_decay_less_eq_one(blt_fn: collections.abc.Callable[..., typing.Any]) -> typing.Any
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.require_buf_decay_less_eq_one

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.require_buf_decay_less_eq_one
```
````

````{py:function} require_output_scale_gt_zero(blt_fn: collections.abc.Callable[..., typing.Any]) -> typing.Any
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.require_output_scale_gt_zero

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.require_output_scale_gt_zero
```
````

````{py:function} blt_pair_from_theta_pair(theta: jax.Array, theta_hat: jax.Array) -> tuple[jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz]
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.blt_pair_from_theta_pair

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.blt_pair_from_theta_pair
```
````

````{py:function} sensitivity_squared(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, n: chex.Numeric) -> float
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.sensitivity_squared

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.sensitivity_squared
```
````

````{py:function} robust_max_error_Gamma_j(omega: jax.Array, theta: jax.Array, n: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_j

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_j
```
````

````{py:function} robust_max_error_Gamma_jk(omega1: jax.Array, theta1: jax.Array, omega2: jax.Array, theta2: jax.Array, n: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_jk

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.robust_max_error_Gamma_jk
```
````

````{py:function} iteration_error(inv_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, i: chex.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.iteration_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.iteration_error
```
````

````{py:function} max_error(inv_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, n: chex.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.max_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.max_error
```
````

````{py:function} limit_max_error(inv_blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_error

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_error
```
````

````{py:function} max_loss(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz, n: jax.Array) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.max_loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.max_loss
```
````

````{py:function} limit_max_loss(blt: jax_privacy.matrix_factorization.buffered_toeplitz.BufferedToeplitz) -> jax.Array
:canonical: jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_loss

```{autodoc2-docstring} jax_privacy.matrix_factorization.buffered_toeplitz.limit_max_loss
```
````
