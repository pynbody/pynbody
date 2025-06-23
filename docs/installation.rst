.. Last checked by AP: 23 June 2025

.. summary How to install pynbody

.. _pynbody-installation:

Pynbody Installation
====================


In brief
--------

Pynbody provides binary distributions for Mac, Linux and (since version 2.2) Windows.
The distributions include support for 64-bit x86 and ARM processors on all three systems.
(However see a special note below about :ref:`windows-on-arm`).

To use the latest release of *pynbody*, you must be using a recent version of Python 3, as supported by
`numpy <http://www.numpy.org>`_ and `scipy <http://www.scipy.org>`_. The standardized
`SPEC0 <https://scientific-python.org/specs/spec-0000/>`_ policy describes the versions of numpy, scipy and Python we aim to support.

To install the latest pre-release version of *pynbody*, use:

.. code-block :: bash

  $ pip install --pre pynbody


To install the latest release version of *pynbody*, use:

.. code-block :: bash

  $ pip install pynbody

This should efficiently install a binary version of pynbody. To install from our bleeding edge, ensure
that your compilers are installed and up to date, and then use:

.. code-block :: bash

  $ pip install git+git://github.com/pynbody/pynbody.git

That's all there is to it, really. But if you have problems or need more help, read on.


Getting python
--------------

**Option 1:** If you administer your own machine, start by downloading and installing the latest version of Python. We generally recommend
downloading directly from the `Python website <http://www.python.org>`_.

You can ``pip install`` directly into your central installation, although it's generally better to use a virtual environment (see below).

**Option 2:** If your don't administer your own machine, but there is a centrally-installed recent version of Python, you can still make your own virtual environment. This is a way of creating a self-contained Python
installation pointing back to the central one. This is done by typing

.. code-block :: bash

  $ python -m venv mypython


where ``mypython`` is the name of the directory you want to create. Then you can activate the environment by typing

.. code-block :: bash

  $ source mypython/bin/activate

You need to activate the environment every time you want to use it. For more information about virtual environments, see the `Python packaging documentation <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments>`_.

**Option 3:** If you do not have administrative access to your machine, and the centrally-installed python is unsuitable (e.g. is
too old), you may want to use a third-party installer such as `Anaconda
<https://www.anaconda.com/download/>`_ which can install to a user folder.
Note that we do not ship binary distributions for the ``conda`` utility, but you can still install pynbody using ``pip`` into your conda environment.




.. _install-pynbody:

Installing and testing for development
--------------------------------------

First, clone the `git repository from Github
<https://github.com/pynbody/pynbody>`_. Pynbody uses `git
<http://git-scm.com/>`_ for development:


0. ``git`` is probably already on your machine -- try typing ``git`` from the shell. If it exists, go to step 2.

1. get the appropriate binary from http://git-scm.com/downloads and install ``git``

2. Clone the git repository:

   .. code-block :: bash

     $ git clone https://github.com/pynbody/pynbody.git


3. Whenever you need the newest version of the repository, run

   .. code-block :: bash

    $ git pull

4. To compile and install, type:

   .. code-block :: bash

      $ cd pynbody
      $ pip install -e .[all]


   If you encounter problems on MacOS, check the :ref:`macos-compilers` section below.

5. Now the package is installed wherever your python packages reside and should be importable from within python.
   The first thing to try is probably running the tests to make sure everything is working:

   .. code-block :: bash

     $ cd tests
     $ # next line is optional: pre-download all test data (otherwise the tests will download them on the fly):
     $ python -c "import pynbody.test_utils as tu; tu.precache_test_data()"
     $ pytest

   If this yields no errors, you are ready to use pynbody in the usual way. If ``pytest`` generates errors and you
   haven't edited the code, please report the error on the `issue tracker <https://github.com/pynbody/pynbody/issues>`_,
   giving as much information as possible. If the ``pytest`` command isn't found, you probably didn't install
   ``pynbody`` with the ``[all]`` option above; you can install ``pytest`` separately with ``pip install pytest``.

6. If you are planning to contribute to the development of pynbody, you should run the tests again before submitting a
   pull request, and ideally find a way to add a test that demonstrates the bug you are fixing. This is not always
   possible, but it is always appreciated. For more information on the testing framework, see the
   `pytest documentation <https://docs.pytest.org/en/latest/>`_.

7. If you are planning to contribute to the development of pynbody, you should also install the pre-commit hooks by
   running the following command:

   .. code-block :: bash

     $ pip install pre-commit
     $ pre-commit install

   The pre-commit hooks will run every time you commit changes to the repository and will check for common formatting
   issues. For more information on the pre-commit hooks, see the `pre-commit documentation <https://pre-commit.com/>`_.

.. note::
   If you plan on joining the development efforts and you are
   unfamiliar with git, we recommend that you spend some time getting
   familiar with it. The `git documentation <http://git-scm.com/doc>`_
   is quite good and it's worth a read through Chapter 3 on
   branching. You may also choose to `fork the repo
   <https://help.github.com/articles/fork-a-repo>`_ if you already
   have a `github <http://github.com>`_ account. And finally, please
   read our `code of conduct <https://github.com/pynbody/pynbody/blob/master/CODE_OF_CONDUCT.md>`_
   for contributors.


.. _macos-compilers:

MacOS compilers
^^^^^^^^^^^^^^^

If you are using MacOS, be aware that Apple's default ``clang`` compiler does not support OpenMP.
Your attempt to install pynbody from source may therefore be unsuccessful, in which case
you need to isntall a different compiler.
We recommend using *gcc* from the `MacPorts <https://www.macports.org/>`_ package.
Once you have installed MacPorts, you can install *gcc* and then use it to install pynbody as
follows:

.. code-block :: bash

  $ sudo port install gcc13
  $ export CC=gcc-mp-13
  $ export CXX=g++-mp-13
  $ pip install -e .[all]


.. _windows-on-arm:

Windows on ARM
^^^^^^^^^^^^^^

Windows-on-ARM is supported, and binary wheels of pynbody are available, but some upstream packages
such as scipy are not providing binary wheels at the time of writing. As such, installation on
Windows-on-ARM requires building some upstream packages from source. Alternatively, you may have
luck installing unofficial pre-built wheels from
`Christoph Gohlke's repository <https://github.com/cgohlke/win_arm64-wheels/releases>`_.


Open your simulation and start analyzing
----------------------------------------

Check out the rest of the :ref:`tutorials section <tutorials>` and
especially the :ref:`data-access` to get going.



Building your own documentation
-------------------------------

You obviously know where to find the documentation since you are
reading it. But if you wanted to build your own flavor of it or if you
want to contribute to the documentation, go to ``docs`` in the root
directory. You will need to install `Sphinx <http://sphinx-doc.org/>`_
and some ancillary packages to build the docs, and this is usually most easily done with
``pip install pynbody[docs]``. Once you have ``sphinx`` installed, you can simply run
``make html`` in the ``docs`` directory to build the html version or
make latexpdf to generate a pdf file, for example. All builds are
found in ``_build``.
