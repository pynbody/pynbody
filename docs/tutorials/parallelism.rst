.. parallelism tutorial

.. _parallelism:

Use of multiple processors by pynbody
=====================================

There are several distinct ways in which parallelization is possible in pynbody.

(1) *Native threading*, using the ``python`` module ``threading``, or in C code,
    the POSIX standard ``pthread`` library. On any modern Mac and Linux machine,
    this "just works". This is mainly used in the SPH module where we have
    gone to some lengths to create algorithms that scale well to large numbers of
    threads.

(2) *OpenMP threading*. This is used especially in Cython routines used for
    interpolation and gravity routines. If you install from a binary distribution
    or have an OpenMP compiler this will also "just work". However, if you are trying
    to build from source on macOS this can cause issues -- :ref:`see below <openmp-fix>`.

(3) *Process parallelism*. Pynbody also exposes a way to share arrays across completely
    separate python processes (on the same machine). This is especially used by
    `tangos <https://github.com/pynbody/tangos>`_ to enable efficient analysis of
    large numbers of halos/galaxies within a single simulation. It is also used internally
    by the ramses loader since loading a ramses file turns out to be an intensive process
    that can usefully be parallelised.

(4) *Partial loading*. Pynbody can also load only part of a simulation, which is especially useful for
    large simulations where you may not have enough memory to load the entire dataset.
    This facility can be helpful in building your own parallel analyses.

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
processes to read files. The optimal number of readers depends on a combination
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


Partially loading snapshots; halo catalogues
--------------------------------------------

Some snapshot formats can be *partially* loaded, for example to read only a region of a large simulation, by
passing ``take=...`` or ``take_region=...`` to :func:`~pynbody.snapshot.load`. Furthermore, if a snapshot is
spanned across multiple files (as for gadget outputs), one can load a single one of these files rather than
the full snapshot.

A halo catalogue can still be used in these cases, but the halo finder may assign
particles to a halo which have not been read from disk. Such a halo is *incomplete*, and trying to access it
raises an :class:`~pynbody.halo.details.particle_indices.IncompleteHaloError` rather than silently returning
too few particles.

This is possible only for halo finders which identify their particles by ID. Some formats instead express
halo membership as positions within the snapshot file, which are meaningless unless every particle is
present. Those catalogues refuse to load against a partially loaded snapshot, raising a
:class:`~pynbody.halo.details.particle_indices.PartialLoadingNotSupportedError`.

Whether a snapshot holds all the particles in its file can be tested directly with
:meth:`~pynbody.snapshot.simsnap.SimSnap.is_partially_loaded`. Note that a view onto a snapshot, such as
``f[:100]`` or ``f.dm``, counts as partially loaded for this purpose.

.. versionadded:: 2.7.0

    A systematic treatment of incomplete halos was added in pynbody 2.7.0.


To find out in advance which halos are affected, use
:meth:`~pynbody.halo.HaloCatalogue.complete_keys`, which returns the subset of
:meth:`~pynbody.halo.HaloCatalogue.keys` that can actually be retrieved. To process only
those halos in a halo catalogue ``h`` that are complete, one would write:

.. sourcecode:: python

  for halo_number in h.complete_keys():
      halo = h[halo_number]
      ...

Individual halos can be tested with :meth:`~pynbody.halo.HaloCatalogue.is_complete`. Finder-calculated
properties remain available for every halo, whether complete or not; it is only access to the particles that
fails. If you need to filter the arrays returned by
:meth:`~pynbody.halo.HaloCatalogue.get_properties_all_halos`, which are in halo index order rather than by
halo number, use :meth:`~pynbody.halo.HaloCatalogue.get_complete_mask`.

.. note::

    Establishing which halos are complete requires the particle lists for all halos, so the methods above call
    :meth:`~pynbody.halo.HaloCatalogue.load_all` for you. Note also that a catalogue built from an array of
    halo numbers, one per particle (such as a ``.grp`` file, or HOP output), cannot detect missing particles,
    since the array covers only the particles which were loaded. Such a catalogue will return those particles
    that it knows about, and raise a warning if you try to access completeness information.



Exposing arrays to other processes
----------------------------------

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
arrays between multiple processes, using shared memory.

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
          pickle.dump(pynbody.array.shared.to_shared_reference(f['pos']), info_file)

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
    pos = pynbody.array.shared.from_shared_reference(shared_array_info)

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
    the ``shared_array_info`` file is worthless -- if you try to call :func:`pynbody.array.shared.from_shared_reference`,
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


Transferring a halo catalogue to another process
------------------------------------------------

.. versionadded:: 2.7.0

Loading a halo catalogue can be expensive, and when analysing a simulation in parallel it is wasteful for
every process to repeat the work. A catalogue can therefore be reduced to a dictionary of numpy arrays and
python primitives, using :meth:`~pynbody.halo.HaloCatalogue.get_portable_state`:

.. sourcecode:: python

    state = h.get_portable_state()

Nothing in this dictionary refers to the halo finder's files, or to the process that created it, so it can be
handed to another process -- for example through the shared memory described above. There, it is turned back
into a working halo catalogue by :meth:`~pynbody.halo.HaloCatalogue.from_portable_state`, which attaches it to
a snapshot that the receiving process has loaded:

.. sourcecode:: python

    h_recreated = pynbody.halo.HaloCatalogue.from_portable_state(state, f)

The recreated catalogue offers the halo membership and finder-calculated properties of the original, but never
touches the halo finder's files. The snapshot it is attached to must present the same particles in the same
order as the one the state was generated from, since halo membership is stored as offsets into the snapshot.

The consumer does not need to know what any individual array in the state is for.
:func:`~pynbody.halo.portable.map_arrays` walks a state and replaces every array in it, so the arrays can be
moved wholesale:

.. sourcecode:: python

    references = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)

and, in the receiving process, turned back into arrays by passing the reference type as ``types``:

.. sourcecode:: python

    state = pynbody.halo.portable.map_arrays(references,
                                             pynbody.array.shared.from_shared_reference,
                                             types=pynbody.array.shared.SharedArrayReference)

For this to work without copying anything, the catalogue's arrays must already be in shared memory. Call
:meth:`~pynbody.snapshot.simsnap.SimSnap.enable_shared_arrays` on the snapshot *before* loading the
catalogue, and its particle index lists -- which are 8 bytes per particle in a halo, so gigabytes for a large
simulation -- are allocated there directly:

.. sourcecode:: python

    f = pynbody.load('gasoline_ahf/g15784.lr.01024')
    f.enable_shared_arrays()      # must come before f.halos()

    h = f.halos()
    state = h.get_portable_state()
    references = pynbody.halo.portable.map_arrays(state, pynbody.array.shared.to_shared_reference)

Whether a snapshot is using shared arrays can be checked with
:meth:`~pynbody.snapshot.simsnap.SimSnap.uses_shared_arrays`. Without it,
:func:`~pynbody.array.shared.to_shared_reference` raises a ``TypeError``, since there is no shared memory to
refer to; the state can still be transferred, but only by copying it (e.g. by pickling it down a pipe, or by
copying each array into shared memory first).

.. versionadded:: 2.7.1

    Halo catalogues allocate their index lists in shared memory when the snapshot asks for it, and the
    portable state preserves that, so no copy is needed to hand a catalogue to another process.

.. warning::

    The process that generated the state owns the shared memory, and releases it when the last reference
    within that process goes away. Keep the state alive for as long as any other process is still using
    the catalogue: the references describe the memory but do not keep it allocated, and holding only the
    originating :class:`~pynbody.halo.HaloCatalogue` is not enough either, since a few of the arrays in a
    state are derived when it is generated rather than being arrays the catalogue itself holds.
