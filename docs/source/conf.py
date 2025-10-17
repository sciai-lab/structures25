# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "STRUCTURES25"
copyright = "2025, sciai-lab"
author = "sciai-lab"

import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
]

# tex config for the math rendering
mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {"packages": {"[+]": ["physics"]}},
}


intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/main/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyscf": ("https://pyscf.org/", None),
    "e3nn": ("https://docs.e3nn.org/en/latest/", None),
}

# Sphinx defaults to automatically resolve *unresolved* labels using all your Intersphinx mappings.
# This behavior has unintended side-effects, namely that documentations local references can
# suddenly resolve to an external location.
# See also:
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_disabled_reftypes
# If this should become a problem, uncomment the following line:
# intersphinx_disabled_reftypes = ["*"]

napoleon_numpy_docstring = False
napoleon_preprocess_types = True

# Hide module names in description unit titles (such as .. function::).
add_module_names = False
# autodoc_default_options = {"members": True, "undoc-members": True, "private-members": True}

templates_path = ["_templates"]
exclude_patterns = []


def setup(app):
    """Add custom CSS overrides."""
    app.add_css_file("theme_override.css")
    app.add_css_file("svg_scaling.css")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_book_theme"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
