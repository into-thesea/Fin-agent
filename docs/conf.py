"""
Fin-Agent Sphinx 文档配置
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# 项目信息
project = 'Fin-Agent'
copyright = '2026, Fin-Agent Team'
author = 'Fin-Agent Team'
release = '4.0.0'

# autodoc 配置
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'private-members': False,
}

# Napoleon 配置
napoleon_google_docstring = True
napoleon_numpy_docstring = False
