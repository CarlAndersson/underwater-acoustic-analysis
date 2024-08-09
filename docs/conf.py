# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "uwacan"
copyright = "2024, Carl Andersson"
author = "Carl Andersson"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.napoleon",
    # "autoapi.extension",
]
add_module_names = False
# autosummary_generate = True
viewcode_line_numbers = True
numpydoc_show_class_members = False  # We run our own template instead
# numpydoc_xref_param_type = False
autosummary_context = {
    "skip_methods": ["__init__"],
    "extra_methods": ["__call__"],
}

# We need to override all these in numpydoc. The default will litter "python:float" and so on.
# numpydoc_xref_aliases = {
#     "None": "None",
#     "bool": "bool",
#     "boolean": "boolean",
#     "True": "True",
#     "False": "False",
#     "list": "list",
#     "tuple": "tuple",
#     "str": "str",
#     "string": "string",
#     "dict": "dict",
#     "float": "float",
#     "int": "int",
#     "callable": "callable",
#     "iterable": "iterable",
#     "sequence": "sequence",
#     "contextmanager": "contextmanager",
#     "namedtuple": "namedtuple",
#     "generator": "generator",
# }
# numpydoc_xref_ignore = "all"
default_role = "py:obj"
# autodoc_default_options = {
#     "special-members": True,
#     # "members": True,
#     "members": True,
# }

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/latest", None),
    "sounddevice": ("https://python-sounddevice.readthedocs.io/en/latest/", None),
}
# autoapi_dirs = ["../uwacan"]
# autoapi_own_page_level = "function"
# autoapi_generate_api_docs = True
# autoapi_options = [
#     "members",
#     "inherited-members",
#     "undoc-members",
#     "show-module-summary",
#     "imported-members",
# ]
# autoapi_ignore = ["*charts*"]
# autoapi_root = "reference/generated"
# autoapi_add_toctree_entry = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
