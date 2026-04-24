# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from typing import Protocol
from unittest import mock
import re

class MockDataclassInstance(Protocol):
  pass

class MockTypeshed(mock.MagicMock):
  def __getattr__(self, name):
    if name == 'DataclassInstance':
      return MockDataclassInstance
    return super().__getattr__(name)

sys.modules['_typeshed'] = MockTypeshed()
sys.path.insert(0, os.path.abspath('..'))

project = 'JAX Privacy'
copyright = '2025, Google DeepMind'
author = 'Google DeepMind'
release = '1.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'myst_nb',
    'sphinx_collections',
    'sphinx.ext.doctest',
]

intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'chex': ('https://chex.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

autodoc_type_aliases = {
    'ArrayLike': 'jax.typing.ArrayLike',
    'ArrayTree': 'chex.ArrayTree',
    'chex.ArrayTree': 'chex.ArrayTree',
    'PydanticDataclass': 'pydantic.PydanticDataclass',
}

autosummary_generate = True

# Configure autodoc settings
autodoc_typehints = 'both'
autoclass_content = 'both'
autodoc_member_order = 'bysource'
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
nb_execution_mode = 'off'
suppress_warnings = ['misc.highlighting_failure']

collections = {
    'examples': {
        'driver': 'copy_folder',
        'source': '../examples',
        'ignore': 'BUILD',
    }
}

def simplify_text(text):
    if not text: return text

    # Very aggressive replacements for complex types
    replacements = [
        # Base regex
        (r'jax\.Array \| numpy\.ndarray \| numpy\.bool \| numpy\.number \| Iterable\[.*?\] \| Mapping\[Any, .*?\]', 'chex.ArrayTree'),
        (r'Array \| ndarray \| bool \| number \| Iterable\[.*?\] \| Mapping\[Any, .*?\]', 'chex.ArrayTree'),

        # HTML regexes
        (r'(?:<a[^>]+>)?<em>(?:jax\.)?Array</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?ndarray</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?bool</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?number</em>(?:</a>)?<em> \| </em><em>Iterable</em><em>\[</em>.*?<em>\] </em><em>\| </em><em>Mapping</em><em>\[</em><em>Any</em><em>, </em>.*?<em>\]</em>', 'chex.ArrayTree'),
        (r'(?:<a[^>]+>)?<em>(?:jax\.)?Array</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?ndarray</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?bool</em>(?:</a>)?<em> \| </em>(?:<a[^>]+>)?<em>(?:numpy\.)?number</em>(?:</a>)?<em> \| </em><em>Iterable</em>\[.*?\]<em> \| </em><em>Mapping</em>\[<em>Any</em>, .*?\]', 'chex.ArrayTree'),
        (r'(?:<a[^>]+>)?<em>(?:jax\.)?Array</em>(?:</a>)? \| (?:<a[^>]+>)?<em>(?:numpy\.)?ndarray</em>(?:</a>)? \| (?:<a[^>]+>)?<em>(?:numpy\.)?bool</em>(?:</a>)? \| (?:<a[^>]+>)?<em>(?:numpy\.)?number</em>(?:</a>)? \| <em>Iterable</em>\[.*?\] \| <em>Mapping</em>\[<em>Any</em>, .*?\]', 'chex.ArrayTree'),

        # Exact string match variations
        ("TypeAliasForwardRef('chex.ArrayTree')", "chex.ArrayTree"),
        ("TypeAliasForwardRef('ArrayTree')", "chex.ArrayTree"),
        ("TypeAliasForwardRef(‘chex.ArrayTree’)", "chex.ArrayTree"),
        ("TypeAliasForwardRef(‘ArrayTree’)", "chex.ArrayTree"),
        ("TypeAliasForwardRef('jax.typing.ArrayLike')", "jax.typing.ArrayLike"),
        ("numpy._typing.ArrayLike", "numpy.typing.ArrayLike"),
        ("<em>TypeAliasForwardRef</em><em>(</em><em>'chex.ArrayTree'</em><em>)</em>", "chex.ArrayTree"),
        ("<em>TypeAliasForwardRef</em>(<em>'chex.ArrayTree'</em>)", "chex.ArrayTree"),
        ("TypeAliasForwardRef('chex.ArrayTree')", "chex.ArrayTree"),
        ("TypeAliasForwardRef(‘chex.ArrayTree’)", "chex.ArrayTree"),

        # Edge cases observed
        (r'<em>(?:jax\.)?Array</em><em> \| </em><em>(?:numpy\.)?ndarray</em><em> \| </em><em>(?:numpy\.)?bool</em><em> \| </em><em>(?:numpy\.)?number</em><em> \| </em><em>Iterable</em><em>\[</em><em>chex\.ArrayTree</em><em>\] </em><em>\| </em><em>Mapping</em><em>\[</em><em>Any</em><em>, </em><em>chex\.ArrayTree</em><em>\]</em>', "chex.ArrayTree"),
        (r'<em>(?:jax\.)?Array</em><em> \| </em><em>(?:numpy\.)?ndarray</em><em> \| </em><em>(?:numpy\.)?bool</em><em> \| </em><em>(?:numpy\.)?number</em><em> \| </em><em>Iterable</em>\[<em>chex\.ArrayTree</em>\]<em> \| </em><em>Mapping</em>\[<em>Any</em>, <em>chex\.ArrayTree</em>\]', "chex.ArrayTree"),
    ]

    for pattern, repl in replacements:
        if 'r' in pattern or '[' in pattern or '\\' in pattern: # Heuristic for regex
            text = re.sub(pattern, repl, text)
        else:
            text = text.replace(pattern, repl)

    # And a final aggressive regex to catch the TypeAliasForwardRef inside an iterable or alone
    text = re.sub(r'TypeAliasForwardRef\([\'"‘]chex\.ArrayTree[\'"’]\)', 'chex.ArrayTree', text)
    text = re.sub(r'<em>TypeAliasForwardRef</em><em>\(</em><em>[\'"‘]chex\.ArrayTree[\'"’]</em><em>\)</em>', 'chex.ArrayTree', text)
    text = re.sub(r'<em>TypeAliasForwardRef</em>\(<em>[\'"‘]chex\.ArrayTree[\'"’]</em>\)', 'chex.ArrayTree', text)

    # Catch any leftover massive unions that end in Iterable/Mapping of chex.ArrayTree
    text = re.sub(r'(?:<a[^>]*>)?<em>(?:jax\.)?Array</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?ndarray</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?bool</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?number</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<em>)?Iterable(?:</em>)?(?:<em>)?\[(?:</em>)?(?:<em>)?chex\.ArrayTree(?:</em>)?(?:<em>)?\](?:</em>)?(?:<em>)?(?: )?\|(?: )?(?:</em>)?(?:<em>)?Mapping(?:</em>)?(?:<em>)?\[(?:</em>)?(?:<em>)?Any(?:</em>)?(?:<em>)?, (?:</em>)?(?:<em>)?chex\.ArrayTree(?:</em>)?(?:<em>)?\](?:</em>)?', 'chex.ArrayTree', text)

    # JAX ArrayLike fallback: Array | ndarray | bool | number | float | int
    text = re.sub(r'(?:<a[^>]*>)?<em>(?:jax\.)?Array</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?ndarray</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?bool</em>(?:</a>)?(?:<em>)? \| (?:</em>)?(?:<a[^>]*>)?<em>(?:numpy\.)?number</em>(?:</a>)?(?:<em>)? \| (?:</em>)?float(?:<em>)? \| (?:</em>)?int', 'jax.typing.ArrayLike', text)

    return text

def simplify_type_hints(app, what, name, obj, options, signature, return_annotation):
    if signature:
        signature = simplify_text(signature)
    if return_annotation:
        return_annotation = simplify_text(return_annotation)
    return signature, return_annotation

def simplify_docstring_text(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        lines[i] = simplify_text(lines[i])

def setup(app):
    app.connect('autodoc-process-signature', simplify_type_hints)
    app.connect('autodoc-process-docstring', simplify_docstring_text)
