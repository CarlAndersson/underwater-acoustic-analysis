Installing
==========
There are different ways to install the package depending on what you intend to do.
We have chosen to use `uv <https://docs.astral.sh/uv/>`_ as our python installer and package tooling.
This guide focuses on how to use uv to setup a development environment.
If you are used to a different python environment, e.g., venv or conda, this package is installable like any other git repository.

- For extending the functionality of this package, see `Installing the package for development`_.
- To write your own code managed in a git repository, see `Including the package as a submodule`_.

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
2. Install the environment with uv

Including the package as a submodule
------------------------------------
1.  Initialize the top repo (git clone or git init)

    .. code-block:: shell

        git init
        uv init

2.  Add the uwacan repo as a submodule

    .. code-block:: shell

        git submodule add git+https://github.com/CarlAndersson/underwater-acoustic-analysis uwacan
        git submodule init

3.  Add the local submodule as an editable dependency

    .. code-block:: shell

        uv add "uwacan @ underwater-acoustics-analysis" --editable

4.  Add some development dependencies for your local code. You can use e.g., ``uv add jupyter`` to add packages.
