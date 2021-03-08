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
import subprocess
import pkg_resources
import datetime

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Summit"
dt = datetime.datetime.today()
year = dt.year
copyright = f"{year}, Summit Authors"
author = "Kobi Felton and Summit Authors"

# The full version, including alpha/beta/rc tags
release = pkg_resources.get_distribution("summit").version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # enables use of jupyter notebooks
    "nbsphinx",
    # enables use of math
    "sphinx.ext.mathjax",
    # enables use of numpy docs strings
    "sphinx.ext.napoleon",
    # enables automatic processing of docstrings
    "sphinx.ext.autodoc",
    # View the code
    "sphinx.ext.viewcode",
    # enables to provide links alias in the project
    "sphinx.ext.intersphinx",
    # read the docs theme
    "sphinx_rtd_theme",
    # Redirects
    "sphinx_reredirects",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Google Colab badge -------------------------------------------------
COLAB = "https://colab.research.google.com/github"


def get_current_git_branch():
    branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
    branch = str(branch).replace("\\n", "").replace("b", "").replace("'", "")
    return branch


def get_colab_base_link(
    user="sustainable-processes", repo="summit", docs_path="docs/source"
):
    branch = get_current_git_branch()
    return f"{COLAB}/{user}/{repo}/blob/{branch}/{docs_path}"


badge_link = "https://colab.research.google.com/assets/colab-badge.svg"
nbsphinx_prolog = r"""
{%% set docname = env.doc2path(env.docname, base=None) %%} 
.. role:: raw-html(raw)
   :format: html

.. |colab_badge| replace:: :raw-html:`<a href="%s/{{ docname }}"><img src="%s" alt="Open in colab"/></a>`
""" % (
    get_colab_base_link(),
    badge_link,
)

# -- Options for NBShpinx ----------------------------------------------------

autoclass_content = "init"


# -- Options for intersphinx output -------------------------------------------

# intersphinx_mapping = dict(GPy=("https://gpy.readthedocs.io/", None))

# -- Options for linkdccode ---------------------------------------------------


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://somesite/sourcerepo/%s.py" % filename


# -- Options for redirects----------------------------------------------------

redirects = {
    "tutorial": "tutorials/intro.html",
    "experiments_benchmarks/new_benchmarks": "../tutorials/new_benchmarks.html",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
# html_theme_options = {
#     #'nosidebar': True,
#     "navigation_with_keys": True,
#     "description": "Tools for optimising chemical processes",
#     "github_banner": True,
#     "github_button": True,
#     "github_repo": "summit",
#     "github_type": "star",
#     "github_user": "sustainable-processes",
#     "page_width": "1095px",
#     "show_relbars": True,
# }
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
