# {py:mod}`jax_privacy.auditing`

```{py:module} jax_privacy.auditing
```

```{autodoc2-docstring} jax_privacy.auditing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BootstrapParams <jax_privacy.auditing.BootstrapParams>`
  - ```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams
    :summary:
    ```
* - {py:obj}`CanaryScoreAuditor <jax_privacy.auditing.CanaryScoreAuditor>`
  - ```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor
    :summary:
    ```
````

### API

`````{py:class} BootstrapParams
:canonical: jax_privacy.auditing.BootstrapParams

```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams
```

````{py:attribute} num_samples
:canonical: jax_privacy.auditing.BootstrapParams.num_samples
:type: int
:value: >
   1000

```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams.num_samples
```

````

````{py:attribute} quantiles
:canonical: jax_privacy.auditing.BootstrapParams.quantiles
:type: numpy.typing.ArrayLike
:value: >
   (0.025, 0.975)

```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams.quantiles
```

````

````{py:attribute} seed
:canonical: jax_privacy.auditing.BootstrapParams.seed
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams.seed
```

````

````{py:method} confidence_interval(num_samples: int = 1000, confidence: float = 0.95, seed: int | None = None) -> jax_privacy.auditing.BootstrapParams
:canonical: jax_privacy.auditing.BootstrapParams.confidence_interval
:classmethod:

```{autodoc2-docstring} jax_privacy.auditing.BootstrapParams.confidence_interval
```

````

`````

`````{py:class} CanaryScoreAuditor(in_canary_scores: collections.abc.Sequence[float], out_canary_scores: collections.abc.Sequence[float])
:canonical: jax_privacy.auditing.CanaryScoreAuditor

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.__init__
```

````{py:method} epsilon_lower_bound(alpha: float, delta: float = 0, one_sided: bool = True) -> float
:canonical: jax_privacy.auditing.CanaryScoreAuditor.epsilon_lower_bound

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.epsilon_lower_bound
```

````

````{py:method} epsilon_raw_counts(min_count: int = 50, delta: float = 0, one_sided: bool = True, *, bootstrap_params: jax_privacy.auditing.BootstrapParams | None = None) -> float | numpy.ndarray
:canonical: jax_privacy.auditing.CanaryScoreAuditor.epsilon_raw_counts

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.epsilon_raw_counts
```

````

````{py:method} tpr_at_given_fpr(fpr: numpy.typing.ArrayLike, *, bootstrap_params: jax_privacy.auditing.BootstrapParams | None = None) -> numpy.ndarray | float
:canonical: jax_privacy.auditing.CanaryScoreAuditor.tpr_at_given_fpr

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.tpr_at_given_fpr
```

````

````{py:method} attack_auroc(*, bootstrap_params: jax_privacy.auditing.BootstrapParams | None = None) -> float | numpy.ndarray
:canonical: jax_privacy.auditing.CanaryScoreAuditor.attack_auroc

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.attack_auroc
```

````

````{py:method} epsilon_from_gdp(alpha: float, delta: float, eps_tol: float = 1e-06) -> float
:canonical: jax_privacy.auditing.CanaryScoreAuditor.epsilon_from_gdp

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.epsilon_from_gdp
```

````

````{py:method} epsilon_one_shot(significance: float, delta: float, one_sided: bool = True) -> float
:canonical: jax_privacy.auditing.CanaryScoreAuditor.epsilon_one_shot

```{autodoc2-docstring} jax_privacy.auditing.CanaryScoreAuditor.epsilon_one_shot
```

````

`````
