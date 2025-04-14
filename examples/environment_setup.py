"""
===================
Environment setup
===================

To run the sample notebooks contained within punchbowl, you must first create a Python environment.
"""

# %%
# To run the sample notebooks contained within punchbowl, you must first create a Python environment. These days, it's recommended to have a local virtual environment for a project rather than a global configuration.
# If using an environment for running Jupyter notebooks (Visual Studio Code, PyCharm, etc), this environment setup may be handled automatically or through UI prompts on notebook execution.
# Note that for a full environment setup, including full PUNCH requirements, see the instructions linked at the `punch-mission README <https://github.com/punch-mission/punch-mission?tab=readme-ov-file#setting-up-the-punch-environment>`_.

# %%
# To set up manually, start within a terminal, and navigate to the directory in which you want to work with notebooks.
# Begin by setting up a virtual environment within the specified directory. Here we'll setup a hidden directory .venv
#
# .. code-block:: shell
#
#   python -m venv .venv
#

# %%
# Next activate the virtual environment by executing the activation script. Note that when returning to this environment in future, you'll also need to activate the environment once again.
#
# .. code-block:: shell
#
#   source .venv/bin/activate
#

# %%
# To get started quickly with punchbowl, one can install a few packages using pip. Other packages of note can be installed in this same way.
#
# .. code-block:: shell
#
#   pip install ipykernel jupyter punchbowl
#

# %%
# From here, one can open a downloaded notebook in an editor of choice (selecting the local virtual environment), or by calling jupyter directly.
#
# .. code-block:: shell
#
#   jupyter downloaded_notebook.ipynb
#
