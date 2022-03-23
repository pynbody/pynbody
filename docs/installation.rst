.. summary How to install pynbody

.. _pynbody-installation:

Pynbody Installation
====================

Nothing makes us happier than a new pynbody user, so we hope that your
installation experience is swift and painless. If you encounter issues
during the installation process, please let us know right away. Once
you complete the installation, don't forget to read the
:ref:`getting-help` section. Finally, if you decide you like pynbody
and you end up using it for your scientific work, please see
:ref:`acknowledging-pynbody`. Enjoy!


In brief
--------

To install the latest release version (which depends only on numpy and scipy), use

``pip install pynbody``

To install from our bleeding edge, use

``pip install git+git://github.com/pynbody/pynbody.git``

If you have problems or need more help, read on.


Getting python
--------------

If you're new to python we recommend using the `Anaconda Python
<https://store.continuum.io/cshop/anaconda/>`_ bundle from Continuum
Analytics that comes with a nice and easy to use package manager
``conda``. They  provide free licenses for academic use, and the default
installation includes all the pakages you require.

As of 2020, Python 2.X is `no longer supported <https://python3statement.org>`_ by the Python developers or by
core modules such as numpy. For this reason, we have also removed support
from pynbody.

If you desparately want to continue using Python 2.7, you can use pip to install old versions
of pynbody, but these are provided without support.

.. _install-pynbody:

Installing pynbody direct from the repository
---------------------------------------------

You can type in your shell:

::

   pip install git+git://github.com/pynbody/pynbody.git

and everything should happen automatically. This will give you
whatever the latest code from the `git repository <https://github.com/pynbody/pynbody>`_.

.. note:: If your distutils are not installed properly and you don't have root permissions, this will fail -- see :ref:`distutils`.

If you don't have ``pip`` or if you want to develop ``pynbody`` here is
how you can do it manually.

First, clone the `git repository from Github
<https://github.com/pynbody/pynbody>`_. Pynbody uses `git
<http://git-scm.com/>`_ for development:

0. ``git`` is probably already on your machine -- try typing ``git`` from the shell. If it exists, go to step 2.

1. get the appropriate binary from http://git-scm.com/downloads and install ``git``

2. ``$ git clone https://github.com/pynbody/pynbody.git``

3. to get the newest from the repository, run ``git pull``.

4. ``$ cd pynbody``

5. ``$ pip install .[all]``

Now the package is installed wherever your python packages reside and should be importable from within python:

6. ``$ cd ~``

7. ``$ python``

8. ``>>> import pynbody``

If this yields no errors, you are done!

.. note::
   If you plan on joining the development efforts and you are
   unfamiliar with git, we recommend that you spend some time getting
   familiar with it. The `git documentation <http://git-scm.com/doc>`_
   is quite good and it's worth a read through Chapter 3 on
   branching. You may also choose to `fork the repo
   <https://help.github.com/articles/fork-a-repo>`_ if you already
   have a `github <http://github.com>`_ account.



Upgrading your installation and testing features or bug-fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use the most recent version from the repository because
a crucial bug has just been fixed, for example, you can easily update
your installation. If you installed using ``pip`` to begin with, simply
do

::

   pip install -I --no-deps git+git://github.com/pynbody/pynbody@master

If you cloned or forked the git repository and installed manually, go
into the top-level ``pynbody`` source directory (the one with
``setup.py`` in it) and do :

::

   git checkout master  # make sure you are on the master branch
   git pull origin master
   pip install .


If you are testing a new feature or a bug fix that resides in a branch
other than ``master`` this procedure is slightly different:

::

   git fetch
   git checkout -b branch origin/branch  # where "branch" will be the name of the branch for bug fix or feature
   pip install .

When you install a new version of the code and you already have a
python session active with ``pynbody`` loaded, you have to (carefully)
reload all of the affected ``pynbody`` modules. The safest is to just
quit and restart the python session if you're not sure.


Open your simulation and start analyzing
----------------------------------------

Check out the rest of the :ref:`tutorials section <tutorials>` and
especially the :ref:`data-access` to get going.



Updating Code
^^^^^^^^^^^^^

Remember that the ``master`` branch is the
code that everyone else receives when they do a fresh clone of the
repository. It is therefore recommended that any development work is
done in a separate branch that is merged back into the main branch
only when it has been satisfactorily checked. See `What a Branch Is
<http://git-scm.com/book/en/Git-Branching-What-a-Branch-Is>`_ and a
primer on `Basic Branching and Merging
<http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging>`_
in the git documentation. This `description of a workflow
<http://sandofsky.com/blog/git-workflow.html>`_ that discusses tidying
up development branches before merging into the master branch is a
good read.

We are in pretty active development stage at the moment, so it's
always a good idea to keep your code updated. If you want to see what
everyone else has been commiting, you can see the `commit history on
the github code site
<https://github.com/pynbody/pynbody/commits/master>`_.


Nose tests
^^^^^^^^^^

The root directory of the pynbody distribution includes a ``nose``
directory, where the unit (nose) tests reside. In order to run them,
you'll need to download the ``testdata`` bundle from the `downloads section
<https://github.com/pynbody/pynbody/releases>`_ of the pynbody site.


Building your own documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You obviously know where to find the documentation since you are
reading it. But if you wanted to build your own flavor of it or if you
want to contribute to the documentation, go to ``docs`` in the root
directory. You will need to install `Sphinx <http://sphinx-doc.org/>`_
to build the docs, and this is usually most easily done with
``easy_install sphinx`` if you have distutils properly
configured. Once you have ``sphinx`` installed, you can simply run
``make html`` in the ``docs`` directory to build the html version or
make latexpdf to generate a pdf file, for example. All builds are
found in ``_build``.
