# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import loco_mujoco  # noqa: E402

# -- Project information -----------------------------------------------------

project = "LocoMuJoCo"
copyright = "2023-2026, Firas Al-Hafez"
author = "Firas Al-Hafez"
release = loco_mujoco.__version__
version = ".".join(release.split(".")[:2])

github_url = "https://github.com/robfiras/loco-mujoco"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_reredirects",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_legacy", "_scripts", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "substitution",
    "attrs_inline",
]
myst_heading_anchors = 3
myst_substitutions = {"release": release}

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
add_module_names = False
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Optional dependencies that are not needed to render the API reference.
autodoc_mock_imports = ["torch", "smplx", "joblib", "bpy", "pxr", "viser", "mjviser", "trimesh", "wandb"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_attr_annotations = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}
intersphinx_timeout = 10

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "LocoMuJoCo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_show_sourcelink = False

html_theme_options = {
    "logo": {"alt_text": "LocoMuJoCo", "text": ""},
    "header_links_before_dropdown": 7,
    "navbar_align": "left",
    "navbar_end": ["version-badge", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "icon_links": [
        {"name": "GitHub", "url": github_url, "icon": "fa-brands fa-github"},
        {
            "name": "Datasets on Hugging Face",
            "url": "https://huggingface.co/datasets/robfiras/loco-mujoco-datasets",
            "icon": "fa-solid fa-database",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "use_edit_page_button": True,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "github-dark",
}

html_context = {
    "github_user": "robfiras",
    "github_repo": "loco-mujoco",
    "github_version": "master",
    "doc_path": "docs",
    "default_mode": "auto",
}

# The landing page has no section sidebar.
html_sidebars = {
    "index": [],
}

# -- Redirects from the pre-1.3 documentation layout ---------------------------

redirects = {
    "source/loco_mujoco.installation": "../getting_started/installation.html",
    "source/loco_mujoco.amass_installation": "../getting_started/amass.html",
    "source/loco_mujoco.api": "../api/index.html",
    "source/loco_mujoco.core": "../api/index.html",
    "source/loco_mujoco.algorithms": "../training/index.html",
    "source/loco_mujoco.task_factories": "../user_guide/task_factories.html",
    "source/loco_mujoco.trajectory": "../user_guide/trajectories.html",
    "source/loco_mujoco.environments.humanoids": "../environments/index.html",
    "source/loco_mujoco.environments.quadrupeds": "../environments/index.html",
    "source/core/terrain": "../../user_guide/terrain.html",
    "source/tutorials/replay_datasets": "../../getting_started/replay_motion.html",
    "source/tutorials/creating_environments": "../../getting_started/first_environment.html",
    "source/tutorials/customizing_environments": "../../user_guide/modular_architecture.html",
    "source/tutorials/creating_new_environments": "../../user_guide/new_robot.html",
    "source/tutorials/real_robot_deployment": "../../sim2real/index.html",
    "source/tutorials/trajectory_interface": "../../user_guide/trajectories.html",
    "source/tutorials/rendering": "../../user_guide/rendering.html",
}

# Set LMJ_DOCS_NO_API=1 for a fast build without the API reference (about ten seconds instead of two minutes).
if os.environ.get("LMJ_DOCS_NO_API"):
    exclude_patterns.append("api")
    autosummary_generate = False

# -- Build hygiene -----------------------------------------------------------

nitpicky = False
suppress_warnings = ["myst.header", "autosummary.import_cycle", "design.card_legacy"]
linkcheck_ignore = [
    r"https://amass\.is\.tue\.mpg\.de.*",
    r"https://smpl.*\.is\.tue\.mpg\.de.*",
    r"https://mano\.is\.tue\.mpg\.de.*",
]
linkcheck_timeout = 20
