{%- extends 'rst.tpl' -%}


{% block input %}
{%- if not cell.source.isspace() -%}
.. code:: python {{"\n"}}
{%- for line in cell.source.split('\n') -%}
    {%- if line -%}
        {%- if line[0].isspace() -%}
            {%- set prefix = "... " -%}
        {%- else  -%}
            {%- set prefix = ">>> " -%}
        {%- endif -%}
    {%- else -%}
        {%- set prefix = "" -%}
    {%- endif -%}
{{ ["\n",prefix,line] | join | indent }}
{%- endfor -%}
{%- endif -%}
{{"\n"}}
{% endblock input %}


{% block stream %}
.. parsed-literal::
    :class: output

{{ output.text | replace("_","\_") | indent }}
{% endblock stream %}


{% block execute_result %}
    {%- if "text/html" in  output.data -%}
.. raw:: html {{"\n"}}
{{ output.data["text/html"]| indent }}
    {%- elif "text/plain" in  output.data -%}
.. parsed-literal::
    :class: output {{"\n"}}
    {{ output.data["text/plain"] | replace("_","\_") | indent }}
    {%- endif -%}
{% endblock execute_result %}

{% block data_text scoped %}
.. parsed-literal::
    :class: output {{"\n"}}
{{ output.text  | replace("_","\_") | indent }}
{% endblock data_text %}

{% block markdowncell scoped %}
{# URL correction in markdown cells #}
{{ cell.source | safe | replace("](../doc/html/","](") | replace(".ipynb",".html")  | markdown2rst }}
{% endblock markdowncell %}
