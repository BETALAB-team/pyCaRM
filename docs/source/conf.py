# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import os
import sys
import tomllib
sys.path.insert(0, os.path.abspath('../..'))

project = 'pyCaRM'
copyright = '2026, Alessio Tollin, Angelo Zarrella'
author = 'Alessio Tollin, Angelo Zarrella'
with open(os.path.join(os.path.dirname(__file__), '../../pyproject.toml'), 'rb') as f:
    release = tomllib.load(f)['project']['version']

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['system.matrix*']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinxawesome_theme'
html_permalinks = False  # rimuove il simbolo ¶

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'inherited-members': False,
    'show-inheritance': True,
    'special-members': False,
    'exclude-members': '__dict__,__weakref__,__dataclass_fields__,__dataclass_params__,__init__',
}

autodoc_class_signature = 'separated'

suppress_warnings = ['ref.python', 'autodoc']

napoleon_use_ivar = False
napoleon_use_param = True