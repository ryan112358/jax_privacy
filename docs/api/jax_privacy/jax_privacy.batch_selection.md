# {py:mod}`jax_privacy.batch_selection`

```{py:module} jax_privacy.batch_selection
```

```{autodoc2-docstring} jax_privacy.batch_selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PartitionType <jax_privacy.batch_selection.PartitionType>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.PartitionType
    :summary:
    ```
* - {py:obj}`BatchSelectionStrategy <jax_privacy.batch_selection.BatchSelectionStrategy>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.BatchSelectionStrategy
    :summary:
    ```
* - {py:obj}`CyclicPoissonSampling <jax_privacy.batch_selection.CyclicPoissonSampling>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling
    :summary:
    ```
* - {py:obj}`BallsInBinsSampling <jax_privacy.batch_selection.BallsInBinsSampling>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.BallsInBinsSampling
    :summary:
    ```
* - {py:obj}`UserSelectionStrategy <jax_privacy.batch_selection.UserSelectionStrategy>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`split_and_pad_global_batch <jax_privacy.batch_selection.split_and_pad_global_batch>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.split_and_pad_global_batch
    :summary:
    ```
* - {py:obj}`pad_to_multiple_of <jax_privacy.batch_selection.pad_to_multiple_of>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.pad_to_multiple_of
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RngType <jax_privacy.batch_selection.RngType>`
  - ```{autodoc2-docstring} jax_privacy.batch_selection.RngType
    :summary:
    ```
````

### API

````{py:data} RngType
:canonical: jax_privacy.batch_selection.RngType
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.RngType
```

````

`````{py:class} PartitionType(*args, **kwds)
:canonical: jax_privacy.batch_selection.PartitionType

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} jax_privacy.batch_selection.PartitionType
```

```{rubric} Initialization
```

```{autodoc2-docstring} jax_privacy.batch_selection.PartitionType.__init__
```

````{py:attribute} INDEPENDENT
:canonical: jax_privacy.batch_selection.PartitionType.INDEPENDENT
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.batch_selection.PartitionType.INDEPENDENT
```

````

````{py:attribute} EQUAL_SPLIT
:canonical: jax_privacy.batch_selection.PartitionType.EQUAL_SPLIT
:value: >
   'auto(...)'

```{autodoc2-docstring} jax_privacy.batch_selection.PartitionType.EQUAL_SPLIT
```

````

`````

````{py:function} split_and_pad_global_batch(indices: numpy.ndarray, minibatch_size: int, microbatch_size: int | None = None) -> list[numpy.ndarray]
:canonical: jax_privacy.batch_selection.split_and_pad_global_batch

```{autodoc2-docstring} jax_privacy.batch_selection.split_and_pad_global_batch
```
````

````{py:function} pad_to_multiple_of(indices: numpy.ndarray, multiple: int) -> numpy.ndarray
:canonical: jax_privacy.batch_selection.pad_to_multiple_of

```{autodoc2-docstring} jax_privacy.batch_selection.pad_to_multiple_of
```
````

`````{py:class} BatchSelectionStrategy
:canonical: jax_privacy.batch_selection.BatchSelectionStrategy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} jax_privacy.batch_selection.BatchSelectionStrategy
```

````{py:method} batch_iterator(num_examples: int, rng: jax_privacy.batch_selection.RngType = None) -> typing.Iterator[numpy.ndarray]
:canonical: jax_privacy.batch_selection.BatchSelectionStrategy.batch_iterator
:abstractmethod:

```{autodoc2-docstring} jax_privacy.batch_selection.BatchSelectionStrategy.batch_iterator
```

````

`````

`````{py:class} CyclicPoissonSampling
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling

Bases: {py:obj}`jax_privacy.batch_selection.BatchSelectionStrategy`

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling
```

````{py:attribute} sampling_prob
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.sampling_prob
:type: float
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling.sampling_prob
```

````

````{py:attribute} iterations
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.iterations
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling.iterations
```

````

````{py:attribute} truncated_batch_size
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.truncated_batch_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling.truncated_batch_size
```

````

````{py:attribute} cycle_length
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.cycle_length
:type: int
:value: >
   1

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling.cycle_length
```

````

````{py:attribute} partition_type
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.partition_type
:type: jax_privacy.batch_selection.PartitionType
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.CyclicPoissonSampling.partition_type
```

````

````{py:method} batch_iterator(num_examples: int, rng: jax_privacy.batch_selection.RngType = None) -> typing.Iterator[numpy.ndarray]
:canonical: jax_privacy.batch_selection.CyclicPoissonSampling.batch_iterator

````

`````

`````{py:class} BallsInBinsSampling
:canonical: jax_privacy.batch_selection.BallsInBinsSampling

Bases: {py:obj}`jax_privacy.batch_selection.BatchSelectionStrategy`

```{autodoc2-docstring} jax_privacy.batch_selection.BallsInBinsSampling
```

````{py:attribute} iterations
:canonical: jax_privacy.batch_selection.BallsInBinsSampling.iterations
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.BallsInBinsSampling.iterations
```

````

````{py:attribute} cycle_length
:canonical: jax_privacy.batch_selection.BallsInBinsSampling.cycle_length
:type: int
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.BallsInBinsSampling.cycle_length
```

````

````{py:method} batch_iterator(num_examples: int, rng: jax_privacy.batch_selection.RngType = None) -> typing.Iterator[numpy.ndarray]
:canonical: jax_privacy.batch_selection.BallsInBinsSampling.batch_iterator

````

`````

`````{py:class} UserSelectionStrategy
:canonical: jax_privacy.batch_selection.UserSelectionStrategy

```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy
```

````{py:attribute} base_strategy
:canonical: jax_privacy.batch_selection.UserSelectionStrategy.base_strategy
:type: jax_privacy.batch_selection.BatchSelectionStrategy
:value: >
   None

```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy.base_strategy
```

````

````{py:attribute} examples_per_user_per_batch
:canonical: jax_privacy.batch_selection.UserSelectionStrategy.examples_per_user_per_batch
:type: int
:value: >
   1

```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy.examples_per_user_per_batch
```

````

````{py:attribute} shuffle_per_user
:canonical: jax_privacy.batch_selection.UserSelectionStrategy.shuffle_per_user
:type: bool
:value: >
   False

```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy.shuffle_per_user
```

````

````{py:method} batch_iterator(user_ids: numpy.ndarray, rng: jax_privacy.batch_selection.RngType = None) -> typing.Iterator[numpy.ndarray]
:canonical: jax_privacy.batch_selection.UserSelectionStrategy.batch_iterator

```{autodoc2-docstring} jax_privacy.batch_selection.UserSelectionStrategy.batch_iterator
```

````

`````
