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

To install from our bleeding edge (which additionally requires cython), use

``pip install git+git://github.com/pynbody/pynbody.git``

If you have problems or need more help, read on.


In detail
---------

Like when you install any software, there is a certain amount of
configuration necessary to get pynbody working.  Fortunately, Python
is installed by default on many operating systems, particularly those
common to astronomers, Mac OS X and Linux.

.. _pynbody-installation-must-haves:

You must have:
^^^^^^^^^^^^^^
  * Python 2.6, 2.7, 3.3 or 3.4. Versions prior to 2.6 and 3.3 are not supported.

  * The standard `numpy` (python numeric arrays) and `scipy` (scientific python) packages

  * If you wish to install from our git repository, you need Cython (at least version 0.20 and preferably
  0.21). If you are going to install from a release tarball, this is not required.

  * Standard development tools, i.e. compilers, libraries etc.
  On Mac OS that's usually Apple's XCode.

  * **Note for Mac OS X 10.8 and 10.9 users:** XCode no longer comes
    with the `gcc` compiler and the `clang` compiler doesn't support
    OpenMP -- if you want to take advantage of some parallelized
    sections of the code, you need to install the OpenMP
    implementation for `clang` from
    http://clang-omp.github.io/#try-openmp-clang *or* install the
    `gcc` compiler using `homebrew <http://brew.sh/>`_.

You will probably also want
^^^^^^^^^^^^^^^^^^^^^^^^^^^

  * `matplotlib`: Plotting functions use `matplotlib`, but you can
    access everything except the built-in plots if `matplotlib` is not
    installed.


  * `ipython <http://ipython.scipy.org/moin/>`_ makes using python
    much easier.  For example it provides command history and tab
    completion.

  * use ``ipython --pylab`` to set matplotlib to interactive mode to
    automatically show the plots you make.  Read
    http://matplotlib.sourceforge.net/users/shell.html#ipython-to-the-rescue
    for details.  Possibly set ``backend : TkAgg`` and ``interactive :
    True`` in your matplotlibrc file, but see the above webpage for
    caveats.

These packages are all standard and well supported.

You might also want
^^^^^^^^^^^^^^^^^^^

  * Installing `h5py <http://code.google.com/p/h5py/>`_ will allow you
    to work with Gadget HDF snapshots. See the :ref:`h5py-ref` below.

.. _distutils:

Setup Python Distutils
----------------------

As the webpage `adding to python
<http://docs.python.org/install/index.html>`_ describes, standard
python packages are installed using distutils. By default, this is
done in the `site-packages` directory that resides wherever python is
installed. If you do not have root permissions, then you will not be
able to install packages there, so you need to tell python to install
them somewhere else (your home directory is a fine option). The steps
below describe how to make this happen, and once you've configured it
properly installing *most* python packages will be a breeze.

1. Create a directory where python packages will be installed.  We
recommend ``${HOME}/python``.

::

   mkdir ~/python


2. Download this `.pydistutils.cfg
<http://pynbody.googlecode.com/files/.pydistutils.cfg>`_ file into
your home directory.

3. Set the ``PYTHONPATH`` environment variable.

::

   setenv PYTHONPATH "${HOME}/python" #put into your .cshrc file OR
   export PYTHONPATH="${HOME}/python" # in your .bashrc file


Install External Packages
-------------------------

Linux
^^^^^

On Linux, use your favourite package manager (like yum or ubuntu) to
install all the packages in one line, for example

::

   yum install matplotlib scipy ipython

This easy way requires root access, so you might have to ask your
system administrator. Alternatively, if you don't have root access or
you want an easier life, we recommend working with `Anaconda Python
<https://store.continuum.io/cshop/anaconda/>`_  which
comes with a neat package manager and can be installed without root
privileges.


Mac OS
^^^^^^

Apple's built-in python can be a pain to work with. The easiest route is to
install a scientific python environment as follows:

Option (a): enthought or anaconda python
"""""""""""""""""""""""""""""""""""""""""

Our currently preferred solution is the `Anaconda Python
<https://store.continuum.io/cshop/anaconda/>`_ bundle from Continuum
Analytics that comes with a nice and easy to use package manager
`conda`. They  provide free licenses for academic use, and the default
installation includes all the pakages you require.

If you are at an academic institution (which is likely the case if you are
installing pynbody) then you can also use  `canopy <http://www.enthought.com/>`_
(formerly Enthought Python Distribution). Go to the `Academic License
<http://www.enthought.com/products/edudownload.php>`_ page and trust them with
your email address to get a download link. See the full `package index
<http://www.enthought.com/products/epdlibraries.php>`_.




Option (b): python's official python
""""""""""""""""""""""""""""""""""""

If for some reason you require more fine-grained control over your
python distribution, you should install the latest offical version of
python (instead of the apple version which came with your box), then
manually download binary versions of `numpy`, `scipy` and
`matplotlib`:

 * Official python .dmg is available here:
   http://www.python.org/download/. If you are going to be sharing  existingcode
   with collaborators, you probably want 2.7.x. However
   pynbody is also now compatible with 3.3 or higher if you are starting on a
   fresh project. (Confused? `See here <https://wiki.python.org/moin/Python2orPython3>`_.)
 * Links to .dmg's for `numpy`, `scipy` and `matplotlib` can be found here:
   `http://scipy.org/install.html#individual-binary-and-source-packages`_.


.. note:: If you are installing numpy/scipy/matplotlib from .dmgs on
 Mac OS >= 10.6, make sure you grab the 10.6 dmgs and *not* the ones
 built for 10.3. If python spits out a menacing error complaining about
 an architecture mismatch, make sure you installed the dmg for the
 correct OS version.


.. _install-pynbody:

Install pynbody
---------------

You can try to type, in your shell:

::

   pip install git+git://github.com/pynbody/pynbody.git

and everything should happen automatically. This will give you
whatever the latest code from the `git repository <https://github.com/pynbody/pynbody>`_.

.. note:: If your distutils are not installed properly and you don't have root permissions, this will fail -- see :ref:`distutils`.

If you don't have `pip` or if you want to develop `pynbody` here is
how you can do it manually.

First, clone the `git repository from Github
<https://github.com/pynbody/pynbody>`_. Pynbody uses `git
<http://git-scm.com/>`_ for development:

0. `git` is probably already on your machine -- try typing ``git`` from the shell. If it exists, go to step 2.

1. get the appropriate binary from http://git-scm.com/downloads and install `git`

2. ``$ git clone https://github.com/pynbody/pynbody.git``

3. to get the newest from the repository, run ``git pull``.

4. ``$ cd pynbody``

5. ``$ python setup.py install``

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
your installation. If you installed using `pip` to begin with, simply
do

::

   pip install -I --no-deps git+git://github.com/pynbody/pynbody@master

If you cloned or forked the git repository and installed manually, go
into the top-level `pynbody` source directory (the one with
``setup.py`` in it) and do :

::

   git checkout master  # make sure you are on the master branch
   git pull origin master
   python setup.py install


If you are testing a new feature or a bug fix that resides in a branch
other than `master` this procedure is slightly different:

::

   git fetch
   git checkout -b branch origin/branch  # where "branch" will be the name of the branch for bug fix or feature
   python setup.py install

When you install a new version of the code and you already have a
python session active with `pynbody` loaded, you have to (carefully)
reload all of the affected `pynbody` modules. The safest is to just
quit and restart the python session if you're not sure.


Open your simulation and start analyzing
----------------------------------------

Check out the rest of the :ref:`tutorials section <tutorials>` and
especially the :ref:`data-access` to get going.



Updating Code
^^^^^^^^^^^^^

Remember that the `master` branch is the
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
configured. Once you have `sphinx` installed, you can simply run
``make html`` in the ``docs`` directory to build the html version or
make latexpdf to generate a pdf file, for example. All builds are
found in ``_build``.



.. _h5py-ref:

Appendix: Notes on Optional Installation of h5py on Mac OS
----------------------------------------------------------

If you installed enthought python (option a), `h5py` is included so
you should be able to work with HDF files immediately. If you used (b)
or (c) and don't want to use HDF files, there's no problem. Otherwise,
read on...

Installing h5py on Mac OS is easy once you have a working HDF5
installation. However **do not install the HDF5 Mac OS binaries
provided on the HDF5 webpage**. For some reason, they simply do not
work properly. Instead download and untar the HDF5
`source <http://www.hdfgroup.org/HDF5/release/obtain5.html>`_.

Assuming you're running on Snow Leopard, use the following command to
configure the package
(`discovered here <http://hdf-forum.184993.n3.nabble.com/Can-t-install-Pytables-something-wrong-with-my-HDF5-installation-td1246998.html>`_):

::

   env ARCHFLAGS="-arch x86_64" LDFLAGS="-arch x86_64" ./configure
   --build=x86_64-apple-darwin10 --target=x86_64-apple-darwin10
   --prefix=/usr/local/hdf5 --with-szlib=/usr/local/src/szip-2.1/szip
   --with-zlib=/usr/local/include,/usr/local/lib }}}

Finally

::

   make sudo make install


Now ``h5py`` will install without much hassle. `Download the source
<http://code.google.com/p/h5py/downloads/list>`_, untar it, and type:

::

   python setup.py configure --hdf5=/usr/local/hdf5/
   python setup.py build
   sudo python setup.py install
