Installing
==========
There are different ways to install the package depending on what you intend to do.
We have chosen to use `pixi <https://pixi.sh>`_ as our python installer and package tooling.
This guide focuses on how to use pixi to setup a development environment.
If you are used to a different python environment, e.g., venv or conda, this package is installable like any other git repository.

- For extending the functionality of this package, see `Installing the package for development`_.
- To write your own code managed in a git repository, see `Including the package as a submodule`_.
- To write some code without version control, see `Using the package without git`_.

.. dropdown:: Installing an editor

    You will need a code editor to work with python.

    .. tab-set::

        .. tab-item:: VS Code

            The most widely used editor is `Visual Studio Code <https://code.visualstudio.com/>`_.
            It has good editing features, strong python language support, and a very nice terminal emulator.
            If you do not intend to use it for writing lots of code, it can be a bit feature-heavy.
            If you go this route, you will not need any other tools.

        .. tab-item:: Spyder

            The `Spyder IDE <https://www.spyder-ide.org/>`_ offers a Matlab-esque environment.
            It is somewhat opinionated with regards to python environments, so it can be difficult to get it working reliably.

        .. tab-item:: Jupyter Lab

            The `Jupyter <https://jupyter.org/>`_ project allows you to write python code and notebooks in the browser.
            The JupyterLab version (instead of jupyter notebook) includes editing of scripts, linked python consoles, and a terminal.
            This creates a good development environment for basic usage.
            You will however need some text editing tool and a terminal to get started.


Installing the package for development
--------------------------------------
1. Clone the git repo
2. Install the environment with pixi

Including the package as a submodule
------------------------------------
1.  Initialize the top repo (git clone or git init)

    .. code-block:: shell

        git init
        pixi init --format pyproject

2.  Add the uwacan repo as a submodule

    .. code-block:: shell

        git submodule add git+https://github.com/CarlAndersson/underwater-acoustic-analysis uwacan
        git submodule init

3.  Add the local submodule as an editable dependency in pixi
    Add this section to ``pyproject.toml``:

    .. code-block:: toml

        [tool.hatch]
            metadata = { allow-direct-references = true }

        [tool.pixi.pypi-dependencies]
        uwacan = { path = "./underwater-acoustic-analysis", editable = true}

4.  Add some development dependencies for your local code. You can use e.g., ``pixi add jupyter`` to add packages,
    or modify the ``[tool.pixi.dependencies]`` table in ``pyproject.toml``.
    Note that any ``uwacan`` dependencies that are not manually added here will be installed from pypi, not conda-forge.
5.  If you intend on making the top level repository a python package (folder or single file), you can leave the config as is and start writing some code.
    Otherwise, you need to remove the current folder as a pixi dependency, otherwise pixi will try to install the current folder in the environment.
    Remove the line ``<project_name> = { path = ".", editable = true }`` in the ``[tool.pixi.pypi-dependencies]`` table.

Using the package without git
-----------------------------
1.  Initialize the top level folder as a pixi project

    .. code-block:: shell

        pixi init

2.  Add uwacan as a dependency
    Until we have a pypi release, we need to link directly to the package repository.
    This only works if we tell the build system to allow direct links.
    Add this section to ``pyproject.toml``

    .. code-block:: toml

        [tool.hatch]
        metadata = { allow-direct-references = true }

        [tool.pixi.pypi-dependencies]
        uwacan = { git = "ssh://git@github.com/CarlAndersson/underwater-acoustic-analysis.git" }

3.  Add some development dependencies for your local code. You can use e.g., ``pixi add jupyter`` to add packages,
    or modify the ``[tool.pixi.dependencies]`` table in ``pyproject.toml``.
    Note that any ``uwacan`` dependencies that are not manually added here will be installed from pypi, not conda-forge.
4.  If you intend on making the top level repository a python package (folder or single file), you can leave the config as is and start writing some code.
    Otherwise, you need to remove the current folder as a pixi dependency, otherwise pixi will try to install the current folder in the environment.
    Remove the line ``<project_name> = { path = ".", editable = true }`` in the ``[tool.pixi.pypi-dependencies]`` table.
