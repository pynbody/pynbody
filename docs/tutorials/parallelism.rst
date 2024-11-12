.. parallelism tutorial

.. _parallelism:

Use of multiple processors by pynbody
=====================================

A large amount of the code in pynbody is designed to run on multiple processors
on a single shared-memory machine. For most people, it's not necessary to worry in
detail about what's going on, but sometimes you may need to understand a bit more,
and this document tries to explain.

There are three distinct ways in which parallelization works in pynbody.

(1) *Native threading*, using the ``python`` module ``threading``, or in C code,
    the POSIX standard ``pthread`` library. On any modern Mac and Linux machine,
    this "just works". This is mainly used in the SPH module where we have
    gone to some lengths to create algorithms that scale well to moderately
    large numbers of cores (16 certainly, often 32) that you'd find on a
    typical analysis workstation.

(2) *OpenMP threading*. This is used especially in Cython routines used for
    interpolation and gravity routines. If you install from a binary distribution
    or have an OpenMP compiler this will also "just work". However, if you are trying
    to build from source on macOS this can cause issues -- :ref:`see below <openmp-fix>`.

(3) *Process parallelism*. Pynbody also exposes a way to share arrays across completely
    separate python processes (on the same machine). This is especially used by
    `tangos <https://github.com/pynbody/tangos>`_ to enable efficient analysis of
    large numbers of halos/galaxies within a single simulation. It is also used internally
    by the ramses loader since loading a ramses file turns out to be an intensive process
    that can usefully be parallelised. It requires shared memory support,
    for which you need the :ref:`the appropriate python module <posix_ipc>`. This
    should be installed automatically if you install pynbody with pip.

.. seealso::

   For more general information about performance in pynbody, see :ref:`this page <performance>`.

Limiting the number of CPUs used by pynbody
--------------------------------------------

In most cases, one just wants the code to be as responsive as possible and
so by default pynbody uses all CPUs on your machine.  However, sometimes this
is not so desirable -- perhaps you need to leave resources for other users,
or for other processes you are running.

Therefore you can limit the number of processors used by pynbody, either
during a session or permanently. Most of the parallelism built into pynbody
is achieved using native or OpenMP threads (cases 1 and 2 above), and the
number of threads can be limited. During a python session, you can type

.. sourcecode:: python

 pynbody.config['number_of_threads'] = 2

which, as an example, limits the number of CPUs in use to 2. To make the
change permanent, create a ``.pynbodyrc`` file in your home directory
with the following section:

.. code-block:: none

  [general]
  number-of-threads: 2

More information on the pynbody configuration system is
:ref:`available here <configuration>`.

.. note::
    The above does not limit the number of CPUs used by the ramses reader,
    which is controlled separately. See :ref:`below <posix_ipc>`.

.. _openmp-fix:

OpenMP-related errors
---------------------

If you attempt to build pynbody from source using a compiler without OpenMP support, you'll
see an error. This most normally happens on macOS, where Apple disable OpenMP by default.

If this happens, please consult the :ref:`macOS installation instructions <macos-compilers>`.

Another problem can arise where different versions of OpenMP are used by different Python
modules. This can result in mysterious errors (see this
`stack overflow post <https://stackoverflow.com/questions/76653505/intelomp-and-llvm-omp-colliding>`_ as
an example. In such a case, your best option is unfortunately
to install all modules from source, being careful to use the same compilers and OpenMP libraries for all
OpenMP modules you will be using.

.. _posix_ipc:

Parallel ramses reader support
------------------------------

The ramses reader speeds up load times by using multiple concurrent
processes to read files. There are two differences between this and the
standard threading techniques used above.

First, for technical reasons related to the
`Python GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_
you need an extra module to make this work. The module is known as
`posix_ipc <https://github.com/osvenskan/posix_ipc/>`_,
and it normally compiles very straight-forwardly on Linux or macOS. It is installed
at the same time as you install pynbody, so long as you installed it in a standard way
with ``pip``. If for some reason you are missing it, you can type ``pip install posix_ipc``.

Second, the optimal number of readers depends on a combination
of CPU and IO performance, which can be especially subtle on network
file system machines. (With lustre, the best number of processes may even be
dependent on how you `striped the data <https://wiki.lustre.org/Configuring_Lustre_File_Striping>`_.)
You should therefore experiment with the number of
processes used for the reading process if optimisation is important to you. This can be done using pynbody's
standard :ref:`configuration <configuration>` system; for instance, create a
``.pynbodyrc`` file in your home directory with the following
section:

.. code-block:: none

   [ramses]
   parallel-read: 4

This specifies 4 processes.

.. note::
 Many systems limit the amount of shared memory available,
 which can cause problems once you enable parallel-reading. See
 :ref:`our separate note on this issue <pitfall_ramses_sharedmem>`.

.. _using_shared_arrays:

Writing your own parallel code
------------------------------

.. versionadded:: 2.0

  Previously, pynbody had a hidden shared memory system that was used internally and by
  `tangos <https://pynbody.github.io/tangos/>`_ to share arrays between processes. This has
  been exposed for general use in pynbody 2.0.

If you want to write parallel processing of large arrays, you can do so using
`Cython <http://cython.org>`_ and OpenMP parallelisation. Since pynbody arrays are
just wrappers around arrays, you can use standard techniques here. The possible complication
is that we have encountered scenarios where OpenMP really dislikes being used across
different python modules, especially if slightly different OpenMP libraries are in use.
You may need to compile pynbody with the same compiler as you are using for your own
code if you run into these issues (:ref:`see below <openmp-fix>`).

For more ambitious analyses you sometimes want to share arrays between
different processes rather than just threads. This is especially important because of
the Python Global Interpreter Lock (GIL) which means that even if you have multiple
threads, only one can be executing Python code at a time.

Pynbody includes the bare bones of a parallel framework that you can use to share
arrays between multiple processes, using shared memory based on `posix_ipc <https://github.com/osvenskan/posix_ipc/>`_.
(An experiment to use Python's in-built shared memory support showed that it is
`insufficiently flexible at this time <https://github.com/pynbody/pynbody/pull/790>`_.)

We strongly recommend that you use pynbody's shared memory support
with an external framework like `tangos <https://github.com/pynbody/tangos>`_, which provides
a much higher-level interface. However, if you want to develop a lower-level parallel approach,
here is a quick template for how you might do it.

On process 1, load the file and any arrays you will need for processing:

.. sourcecode:: python

      import pickle
      import pynbody

      # Load the file
      f = pynbody.load('gasoline_ahf/g15784.lr.01024')

      # Indicate that you will be using shared memory
      f.enable_shared_arrays()

      # Now let's share the position array with another process.
      # We will do this by writing out a short file with information about the shared
      # array, that we will then load in the other process. Note this information could
      # just as well be passed over a pipe or socket (though obviously is only valid on
      # the same machine).

      with open('shared_array_info', 'wb') as info_file:
          pickle.dump(pynbody.array.shared.pack(f['pos']), info_file)

You can verify that ``shared_array_info`` is just a small file. The actual data is stored in shared
memory, which on linux can be seen in ``/dev/shm/``. The pynbody shared memory is always named
``/dev/shm/pynbody-<random string>``. (On MacOS it does not seem to be possible
to easily see shared memory segments.)

Now keep that Python interpreter open, and open a second interpreter to access the position array:

.. sourcecode:: python

    import pickle
    import pynbody

    # Load the shared array information
    with open('shared_array_info', 'rb') as f:
        shared_array_info = pickle.load(f)

    # Now we can load the shared array
    pos = pynbody.array.shared.unpack(shared_array_info)

    # Now we can use pos as if it were a normal numpy array
    print(pos)

    # Modifications to pos in any process get seen across all others
    pos += 1

At the end, we modified the position array. You can verify that the change is reflected in the
first process, because they are using the same physical memory.

At this very low level, all responsibility for synchronizing access to the shared memory is
on you. Again, for most purposes, we recommend using a higher-level framework like
`tangos <https://github.com/pynbody/tangos>`_, which hides these details away.

.. note::
    Understanding the lifetime of shared memory can be tricky.

    The shared array will only get deleted when the first process is closed. After this point,
    the ``shared_array_info`` file is worthless -- if you try to call :func:`pynbody.array.shared.unpack`,
    you will get a `SharedArrayNotFound` exception. That said, the actual memory continues to be allocated
    until the last process using it is closed, so processes that already have a handle on the shared array
    will continue to be able to access it. (This is a feature of UNIX shared memory, not pynbody.)

    If the process that created the shared memory is killed nicely, pynbody will try to clear up the
    shared memory. However if it is killed with a ``kill -9`` or similar, it is not possible to free
    the shared memory. This is generally not a huge problem because the memory will just get paged out to
    disk and then finally freed on the next reboot. However, on some linux systems there is a limit
    to the total amount of shared memory that can be allocated, and so e.g. on PBS systems you may need
    to clear up after yourself if a job is killed by the scheduler. You can do this by hand
    using ``rm -f /dev/shm/pynbody-*``. (Even if other users have active shared memory segments,
    this will only delete your own.)
