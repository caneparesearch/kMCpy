# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
import tomllib

sys.path.insert(0, os.path.abspath("../../"))
pyproject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyproject.toml"))
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

# -- Project information -----------------------------------------------------

project = "kMCpy"
current_year = datetime.now().year
copyright = f"2022-{current_year}, Canepa Research Lab at University of Houston and DENG Group at NUS"
author = "Zeyu Deng"

version = pyproject_data["project"]["version"]
release = version

src_dir = os.path.abspath(os.path.dirname(__file__))
# version_file = os.path.join("../../kmcpy", "_version.py")
# with io_open(version_file, mode="r") as fd:
#     exec(fd.read())


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["walkthrough/.ipynb_checkpoints/*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

nbsphinx_prompt_width = "0"

master_doc = "index"


source_suffix = [".rst", ".md"]
