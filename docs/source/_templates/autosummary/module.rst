{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-index:

{% if classes %}
Classes
-------
.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in classes %}
   {{ item }}
   {% endfor %}
{% endif %}

{% if functions %}
Functions
---------
.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in functions %}
   {{ item }}
   {% endfor %}
{% endif %}

{% if exceptions %}
Exceptions
----------
.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in exceptions %}
   {{ item }}
   {% endfor %}
{% endif %}

{% if attributes %}
Attributes
----------
.. autosummary::
   :toctree:
   :nosignatures:

   {% for item in attributes %}
   {{ item }}
   {% endfor %}
{% endif %}
