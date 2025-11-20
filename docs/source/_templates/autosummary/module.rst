{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :no-members:


.. rubric:: Public API

{% if classes %}
Classes
-------
.. autosummary::
   :nosignatures:

   {% for item in classes %}
   {{ item }}
   {% endfor %}
{% endif %}

{% if functions %}
Functions
---------
.. autosummary::
   :nosignatures:

   {% for item in functions %}
   {{ item }}
   {% endfor %}
{% endif %}

Reference
---------

{% if classes %}
{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :show-inheritance:
{% endfor %}
{% endif %}

{% if functions %}
{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
{% endif %}
