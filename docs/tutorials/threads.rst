.. threads tutorial

.. _threads:

Use of multiple processors by pynbody
=====================================

A large amount of the code in pynbody is designed to run on multiple processors
on a single shared-memory machine. There are three distinct ways in which
parallelization works in pynbody.

(1) *Native threading*, using the ``python`` module ``threading``, or in C code,
    the POSIX standard ``pthread`` library. On any modern Mac and Linux machine,
    this "just works". This is mainly used in the SPH module where we have
    gone to some lengths to create algorithms that scale well to moderately
    large numbers of cores (16 certainly, often 32) that you'd fine on a
    typical analysis workstation.

(2) *OpenMP threading*. This is used by a large number of parts of the library,
    for example some of the halo analysis code, where simple, inexpensive
    parallelization is possible. If you have an OpenMP compiler this will also
    "just work". However, you might not do - :ref:`see below <openmp-fix>`.

(3) *Process threading*. This is currently used only by the ramses loader to
    get around problems with the python GIL. It requires shared memory support,
    for which you need the :ref:`the appropriate python module <posix_ipc>`.


In cases (2) and (3), you may have to do some work to get everything working.
For (1), it should just work out the box.



Limiting the number of CPUs used by pynbody
--------------------------------------------

In most cases, one just wants the code to be as responsive as possible and
so by default pynbody uses all CPUs on your machine.  However, sometimes this
is not so desirable - perhaps you need to leave resources for other users,
or for other processes you are running.

Therefore you can limit the number of processors used by pynbody, either
during a session or permanently. During a python session, you can type

.. sourcecode:: python

 pynbody.config['number_of_threads']=2

which, as an example, limits the number of CPUs in use to 2. To make the
change permanent, create a ``.pynbodyrc`` file in your home directory
with the following section:

.. code-block:: none

  [general]
  number-of-threads: 2

More information on the pynbody configuration system is
:ref:`available here <configuration>`.

.. _openmp-fix:

Checking and fixing OpenMP support
----------------------------------

If pynbody is installed on a machine without OpenMP support, it'll normally
throw up some complaint during the installation procedure. The installation
will succeed, but many parallel procedures will run only on one core.

To check if this has happened to you, try the following commands

.. sourcecode:: ipython

 In [1]: import pynbody

 In [2]: pynbody.config['number_of_threads']
 Out[2]: 8

So far, we can see that the main pynbody configuration has seen more than
one core on the machine. But let's see how many cores OpenMP can see:

.. sourcecode:: ipython

 In [3]: pynbody.openmp.get_cpus()
 Out[3]: 1

If this number is larger than one, OpenMP support is enabled and you don't
need to do anything more. If it's 1, however, OpenMP support is disabled
and you're losing out.

This happens when, during setup, pynbody could not find the OpenMP library.
The most common scenario where this happens is on a Mac - Apple's default
compilers do not support OpenMP. A fast solution is to download
the latest GNU compilers http://hpc.sourceforge.net, install them, and
point the setup procedure to gcc. At your shell, something like this would do
it -

.. sourcecode:: bash

  $ export CC='/usr/local/bin/gcc'
  $ cd pynbody
  $ rm -rf build/
  $ pip install .

Now, with luck, you'll see that OpenMP is enabled.

.. sourcecode:: ipython

 In [1]: import pynbody

 In [2]: pynbody.openmp.get_cpus()
 Out[2]: 8


If you're still having trouble, you can try
:ref:`asking us for further help <getting-help>`.


.. _posix_ipc:

Parallel ramses reader support
------------------------------

The ramses reader speeds up load times by using multiple concurrent
processes to read files. There are two differences between this and the
standard threading techniques used above.

First, the correct number of processes to use may not be the number of CPUs
on your machine - it's tied more to IO performance than to raw computing power.
Second, for technical reasons related to the
`Python GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_
you need an extra module to make this work. The module is known as ``posix_ipc``,
and it normally compiles very straight-forwardly on Linux or Mac OS. At your
standard command prompt just type:

.. sourcecode:: bash

  $ easy_install posix_ipc

and you should be done. There's not even any need to reinstall pynbody.

As explained above, you may well want to change the number of
processes used for the reading process. This can be done using pynbody's
standard :ref:`configuration <configuration>` system; for instance, create a
``.pynbodyrc`` file in your home directory with the following
section:

.. code-block:: none

   [ramses]
   parallel-read: 4

This specifies 4 processes. You can experiment with this number to
see what works best on your system, which depends on a combination
of CPU and IO performance.

.. note::
 Many systems limit the amount of shared memory available,
 which can cause problems once you enable parallel-reading. See
 :ref:`our separate note on this issue <pitfall_ramses_sharedmem>`.
