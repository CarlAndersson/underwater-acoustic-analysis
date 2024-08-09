{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods -%}
   {% if methods -%}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: generated

   {% for item in methods -%}
      {% if item not in skip_methods %}
      ~{{ name }}.{{ item }}
      {%- endif %}
   {%- endfor %}
   {% for item in extra_methods -%}
      {% if item in all_methods %}
      ~{{ name }}.{{ item }}
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
