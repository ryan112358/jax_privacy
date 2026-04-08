{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

.. rubric:: Batch Selection Strategy API

.. autoclass:: BatchSelectionStrategy
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: autosummary/module.rst

{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
   :toctree:

{% for item in attributes %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:

{% for item in functions %}
   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: autosummary/class.rst

{% for item in classes %}
{% if item != 'BatchSelectionStrategy' %}
   {{ item }}
{% endif %}
{% endfor %}
{% endif %}
{% endblock %}
