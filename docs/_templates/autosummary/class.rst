{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. role:: python(code)
   :language: python

.. default-literal-role:: python

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods -%}
   {% if methods -%}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: generated

   {% for item in methods -%}
      {% if item not in skip_methods %}
      {% if item not in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endif %}
   {%- endfor %}
   {% for item in extra_methods -%}
      {% if item in all_methods %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}

   .. rubric:: {{ _('Inherited methods') }}

   .. autosummary::
      :toctree: generated

   {% for item in methods -%}
      {% if item not in skip_methods %}
      {% if item in inherited_members %}
      ~{{ name }}.{{ item }}
      {%- endif %}
      {%- endif %}
   {%- endfor %}

   {%- endif %}
   {%- endblock %}

   {% block attributes -%}
   {% if attributes -%}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
