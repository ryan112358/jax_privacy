# {py:mod}`jax_privacy.accounting.accountants`

```{py:module} jax_privacy.accounting.accountants
```

```{autodoc2-docstring} jax_privacy.accounting.accountants
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DpAccountantConfig <jax_privacy.accounting.accountants.DpAccountantConfig>`
  - ```{autodoc2-docstring} jax_privacy.accounting.accountants.DpAccountantConfig
    :summary:
    ```
* - {py:obj}`RdpAccountantConfig <jax_privacy.accounting.accountants.RdpAccountantConfig>`
  - ```{autodoc2-docstring} jax_privacy.accounting.accountants.RdpAccountantConfig
    :summary:
    ```
* - {py:obj}`PldAccountantConfig <jax_privacy.accounting.accountants.PldAccountantConfig>`
  - ```{autodoc2-docstring} jax_privacy.accounting.accountants.PldAccountantConfig
    :summary:
    ```
````

### API

`````{py:class} DpAccountantConfig
:canonical: jax_privacy.accounting.accountants.DpAccountantConfig

```{autodoc2-docstring} jax_privacy.accounting.accountants.DpAccountantConfig
```

````{py:method} create_accountant() -> dp_accounting.PrivacyAccountant
:canonical: jax_privacy.accounting.accountants.DpAccountantConfig.create_accountant
:abstractmethod:

```{autodoc2-docstring} jax_privacy.accounting.accountants.DpAccountantConfig.create_accountant
```

````

`````

`````{py:class} RdpAccountantConfig
:canonical: jax_privacy.accounting.accountants.RdpAccountantConfig

Bases: {py:obj}`jax_privacy.accounting.accountants.DpAccountantConfig`

```{autodoc2-docstring} jax_privacy.accounting.accountants.RdpAccountantConfig
```

````{py:attribute} orders
:canonical: jax_privacy.accounting.accountants.RdpAccountantConfig.orders
:type: collections.abc.Sequence[int]
:value: >
   'field(...)'

```{autodoc2-docstring} jax_privacy.accounting.accountants.RdpAccountantConfig.orders
```

````

````{py:method} create_accountant() -> dp_accounting.rdp.RdpAccountant
:canonical: jax_privacy.accounting.accountants.RdpAccountantConfig.create_accountant

````

`````

`````{py:class} PldAccountantConfig
:canonical: jax_privacy.accounting.accountants.PldAccountantConfig

Bases: {py:obj}`jax_privacy.accounting.accountants.DpAccountantConfig`

```{autodoc2-docstring} jax_privacy.accounting.accountants.PldAccountantConfig
```

````{py:attribute} value_discretization_interval
:canonical: jax_privacy.accounting.accountants.PldAccountantConfig.value_discretization_interval
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.accounting.accountants.PldAccountantConfig.value_discretization_interval
```

````

````{py:method} create_accountant() -> dp_accounting.pld.PLDAccountant
:canonical: jax_privacy.accounting.accountants.PldAccountantConfig.create_accountant

````

`````
