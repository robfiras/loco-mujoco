# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = 'LocoMuJoCo'
copyright = '2024, Firas Al-Hafez'
author = 'Firas Al-Hafez'
release = 'v0.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', "m2r2", 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_module_names = False
autodoc_member_order = 'groupwise'
source_suffix = [".rst", ".md"]
napoleon_google_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'css/table.css',
]

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'logo_only': True,
    'display_version': False,
}

html_logo = "https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a"
