import os
import csv
from importlib.metadata import version as get_version

from packaging.version import Version

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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "punchbowl"
copyright = "2024, PUNCH Science Operations Center"
author = "PUNCH Science Operations Center"

# The full version, including alpha/beta/rc tags
release: str = get_version("punchbowl")
version: str = release
_version = Version(release)
if _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"


# -- Metadata documentation ---------------------------------------------------

def copy_and_truncate_csv(src_path, dest_path, columns, filter_column, filter_value, default_key):
    with open(src_path, mode='r', newline='') as src_file:
        reader = csv.DictReader(src_file)
        filtered_rows = [
            {
                **{col: row[col] for col in columns},
                'COMMENT': row['DEFAULT'] if not row['COMMENT'] else row['COMMENT']
            } if row[filter_column] == filter_value else None
            for row in reader
        ]
        filtered_rows = [row for row in filtered_rows if row]  # Remove None entries

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, mode='w', newline='') as dest_file:
        writer = csv.DictWriter(dest_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(filtered_rows)

filter_column = 'TYPE'
filter_value = 'keyword'

src_csv_path = os.path.abspath(os.path.join('../../punchbowl/data/data/', 'omniheader.csv'))
dest_csv_path = os.path.abspath(os.path.join('./data/', 'omniheader_select.csv'))
columns_to_include = ['KEYWORD', 'COMMENT']

copy_and_truncate_csv(src_csv_path, dest_csv_path, columns_to_include, filter_column, filter_value, 'DEFAULT')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ["autoapi.extension",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx_favicon",
              "nbsphinx",
              "sphinxcontrib.mermaid",
              "IPython.sphinxext.ipython_console_highlighting"]

nbsphinx_requirejs_path = ""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/punch-mission/punchbowl",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
    "logo": {
        "text": "punchbowl",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    }
}
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "punch-mission",
    "github_repo": "punchbowl",
    "github_version": "main",
    "doc_path": "docs/source/",
}


autoapi_dirs = ["../../punchbowl"]

favicons = ["favicon.ico"]

mermaid_verbose = True
