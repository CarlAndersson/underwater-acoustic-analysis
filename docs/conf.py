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
import sys
sys.path.insert(0, "ext")
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "default_literal_role",
    "sphinx_design",
    "sphinx_copybutton",
]
add_module_names = False
# autosummary_generate = True
viewcode_line_numbers = True
numpydoc_show_class_members = False  # We run our own template instead

autosummary_context = {
    "skip_methods": [
        "__init__", # we document init in class
        # skip all inherited methods from mapping
        "clear",
        "get",
        "items",
        "keys",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "values",
    ],
    "extra_methods": ["__call__"],
}

default_role = "py:obj"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/latest", None),
    "sounddevice": ("https://python-sounddevice.readthedocs.io/en/latest", None),
    "soundfile": ("https://python-soundfile.readthedocs.io/en/latest", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
