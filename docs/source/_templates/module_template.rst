{{ name | escape | underline}}

..
   Also see https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc

.. automodule:: {{ fullname }}
   :members:
   :private-members:
   :special-members:
   :exclude-members: __weakref__, __hash__

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: module_template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
