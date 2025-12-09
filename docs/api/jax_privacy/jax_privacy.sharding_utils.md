# {py:mod}`jax_privacy.sharding_utils`

```{py:module} jax_privacy.sharding_utils
```

```{autodoc2-docstring} jax_privacy.sharding_utils
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`flatten_with_zero_redundancy <jax_privacy.sharding_utils.flatten_with_zero_redundancy>`
  - ```{autodoc2-docstring} jax_privacy.sharding_utils.flatten_with_zero_redundancy
    :summary:
    ```
* - {py:obj}`local_reshape_add <jax_privacy.sharding_utils.local_reshape_add>`
  - ```{autodoc2-docstring} jax_privacy.sharding_utils.local_reshape_add
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PyTree <jax_privacy.sharding_utils.PyTree>`
  - ```{autodoc2-docstring} jax_privacy.sharding_utils.PyTree
    :summary:
    ```
* - {py:obj}`PartitionSpecPyTree <jax_privacy.sharding_utils.PartitionSpecPyTree>`
  - ```{autodoc2-docstring} jax_privacy.sharding_utils.PartitionSpecPyTree
    :summary:
    ```
````

### API

````{py:data} PyTree
:canonical: jax_privacy.sharding_utils.PyTree
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.sharding_utils.PyTree
```

````

````{py:data} PartitionSpecPyTree
:canonical: jax_privacy.sharding_utils.PartitionSpecPyTree
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} jax_privacy.sharding_utils.PartitionSpecPyTree
```

````

````{py:function} flatten_with_zero_redundancy(abstract_array: jax.Array) -> jax.Array
:canonical: jax_privacy.sharding_utils.flatten_with_zero_redundancy

```{autodoc2-docstring} jax_privacy.sharding_utils.flatten_with_zero_redundancy
```
````

````{py:function} local_reshape_add(x: jax.Array, y: jax.Array) -> jax.Array
:canonical: jax_privacy.sharding_utils.local_reshape_add

```{autodoc2-docstring} jax_privacy.sharding_utils.local_reshape_add
```
````
