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


If you are already a regular python/numpy/scipy user
----------------------------------------------------
Either:

1. If you have have `setuptools <http://pypi.python.org/pypi/setuptools>`_ installed, just type ``easy_install pynbody``. 

.. note:: If your distutils are not installed properly and you don't have root permissions, this will fail -- see :ref:`distutils`. 

Or:

2. Follow the instructions in the :ref:`install-pynbody` section

If you are new to python or you are missing some of the basic packages
(see the `must-haves` below), read on...


Before you start
----------------

Like when you install any software, there is a certain amount of
configuration necessary to get pynbody working.  Fortunately, Python
is installed by default on many operating systems, particularly those
common to astronomers, Mac OS X and Linux.


You must have:
^^^^^^^^^^^^^^
  * Python 2.5, 2.6 or 2.7. We will support 3.x once `scipy` and
    `matplotlib` do.

  * The standard `numpy` (python numeric arrays) package.

You will probably also want
^^^^^^^^^^^^^^^^^^^^^^^^^^^

  * `matplotlib`: Plotting functions use `matplotlib`, but you can
    access everything except the built-in plots if `matplotlib` is not
    installed.

  * Some functionality requires `scipy <http://new.scipy.org/>`_, but
    you can access most functions even if `scipy` is not installed.

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

  * Amiga Halo Finder.  Installation instructions are on
    `HaloCatalogue <http://code.google.com/p/pynbody/wiki/HaloCatalogue>`_

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
system administrator.

It is not hard to perform from-source installations of these packages
if you don't have administrative privileges or a helpful sys admin.
Grab the source from the following sourceforge sites appropriate to
your version of python:

 * `numpy <http://sourceforge.net/projects/numpy/files/>`_

 * `scipy <http://sourceforge.net/projects/scipy/files/>`_

 * `matplotlib <http://sourceforge.net/projects/matplotlib/files/>`_ 

The three packages are standard for nearly all scientific computation
in python, so it makes sense for them to be installed at the system
level.  However, if you sys admin is unhelpful, now that you've set up
distutils, you just have to ``cd`` in each directory and type: 

::

   python setup.py install 


Mac OS
^^^^^^

Choose one of the three options below.  Any of the three options can
be made to work. The first is easiest, the last is hardest; so if you
don't have a strong reason to do otherwise, we'd recommend option (a).

Option (a): enthought python 
""""""""""""""""""""""""""""

If you are at an academic institution (which is likely the case if you
are installing pynbody) then the `Enthought python bundle
<http://www.enthought.com/>`_ is the simplest way of getting
everything you need and more. Go to the `Academic License
<http://www.enthought.com/products/edudownload.php>`_ page and trust
them with your email address to get a download link. It installs
*everything* you need including the core python, numpy, scipy,
matplotlib and other libraries. See the full
`package index <http://www.enthought.com/products/epdlibraries.php>`_.

Option (b): python's official python
""""""""""""""""""""""""""""""""""""

If for some reason you require more fine-grained control over your
python distribution, you should install the latest offical version of
python (instead of the apple version which came with your box), then
manually download binary versions of `numpy`, `scipy` and
`matplotlib`:

 * Official python .dmg is available here:
   http://www.python.org/download/. You want 2.7.x, not 3.x.
 * Links to .dmg's for `numpy
   <http://sourceforge.net/projects/numpy/files/>`_ and `scipy
   <http://sourceforge.net/projects/scipy/files/>`_ are here:
   http://new.scipy.org/download.html.
 * .dmg for matplotlib is here:
   http://sourceforge.net/projects/matplotlib/files/matplotlib/. As of
   this writing there is no dmg for Mac OS > 10.3, but there are many
   other ways of obtaining matplotlib without compiling from source --
   see the `matplotlib Mac OS install notes
   <http://matplotlib.sourceforge.net/faq/installing_faq.html#os-x-notes>`_


.. note:: If you are installing numpy/scipy/matplotlib from .dmgs on
 Mac OS >= 10.6, make sure you grab the 10.6 dmgs and *not* the ones
 built for 10.3. If python spits out a menacing error complaining about
 an architecture mismatch, make sure you installed the dmg for the
 correct OS version.

Option (c): python that came with your Mac
""""""""""""""""""""""""""""""""""""""""""

This is in general not the preferred python solution. 

.. note:: As of December 2011, the scipy superpack no longer appears
 to be maintained.* Therefore should you wish to use the framework
 python that ships with your mac, you'll need to manually compile
 `scipy` and `matplotlib`, and possibly an updated version of
 `numpy`. This can be painful, so we advise option (a) or option (b).


.. _install-pynbody:

Install pynbody
---------------

You should be able to type, in your shell, `easy_install pynbody` and
everything will happen automatically. 

.. note:: If your distutils are not installed properly and you don't have root permissions, this will fail -- see :ref:`distutils`. 

If you don't have `easy_install` or want to do it manually:

1. Download most recent version from `Downloads section <http://code.google.com/p/pynbody/downloads/list>`_ or scroll down to find out how to :ref:`get the most recent version from the repository <repository_section>`

2. Uncompress:  ``tar zxf pynbody.VER-NUMalpha.tar.gz``

3. Enter directory: ``cd pynbodybeta.01``

4. ``python setup.py install``

Open your simulation and start analyzing
----------------------------------------

Check out the rest of the :ref:`tutorials section <tutorials>` and
especially the :ref:`data-access` to get going.


.. _repository_section: 

Staying on the bleeding edge
----------------------------

To get the most recent code, you can check the code out of our Google
Code source repository.  Pynbody uses `git <http://git-scm.com/>`_ for 
development. 

0. `git` is probably already on your machine -- try typing ``git`` from the shell. If it exists, go to step 2.

1. Install `git` 
 
	get the appropriate binary from http://git-scm.com/downloads and install

2. see the `instructions on Google Code <https://code.google.com/p/pynbody/source/checkout>`_. 

3. to get the newest from the repository, run ``git pull``.

If you plan on joining the development efforts and you are unfamiliar with git, we recommend
that you spend some time getting familiar with it. The `git documentation <http://git-scm.com/doc>`_ is quite good and it's worth a read through Chapter 3 on branching.  


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
everyone else has been commiting, you can see the `graph on the google
code site <https://code.google.com/p/pynbody/source/list>`_.


Nose tests
^^^^^^^^^^

The root directory of the pynbody distribution includes a ``nose``
directory, where the unit (nose) tests reside. In order to run them, you'll need to download the ``testdata`` bundle from the `downloads section <https://code.google.com/p/pynbody/downloads/list>`_ of the pynbody site. 


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
