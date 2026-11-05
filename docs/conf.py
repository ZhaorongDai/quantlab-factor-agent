"""Sphinx configuration of the quantlab-factor-agent documentation site."""

import importlib.metadata

project = "quantlab-factor-agent"
language = "en"
author = "Zhaorong Dai"
copyright = "2026, Zhaorong Dai"
try:
    release = importlib.metadata.version("quantlab-factor-agent")
except importlib.metadata.PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# Local notes kept beside the pages (design.md) are not part of the site.
exclude_patterns = ["_build", "design.md", "README.md"]
templates_path = ["_templates"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"release": release}

autosummary_generate = True
autosummary_generate_overwrite = True

autodoc_default_options = {"members": True, "inherited-members": False, "show-inheritance": True}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks = set()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "quantlab": ("https://zhaorongdai.github.io/quantlab/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"quantlab-factor-agent {release}"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/ZhaorongDai/quantlab-factor-agent",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {"default_mode": "auto"}

# Keep package names in signatures and page titles, not in every table entry.
add_module_names = False
html_show_sourcelink = False
html_sidebars = {"**": ["sidebar-nav-bs"]}
