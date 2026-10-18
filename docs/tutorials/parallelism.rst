.. parallelism tutorial

.. _parallelism:

Parallelism in pynbody
======================

Pynbody uses more than one CPU in several distinct ways, each described in a section below.

* :ref:`Calculation parallelism <calculation_parallelism>`: the computationally intensive routines -- building a
  KD-tree, SPH smoothing, rendering images, geometric filters, gravity and centring -- run on several threads, by
  default one per CPU.

* :ref:`Reader parallelism <parallel_io>`: RAMSES and HDF5 outputs are read and decoded on threads.

* :ref:`Limiting parallelism <limiting_threads>`: how to use fewer CPUs, either permanently or for a while; in
  particular, when you run threads of your own.

* :ref:`Free-threaded analyses <free_threading>`: on the free-threaded build of Python, your own analysis can be
  spread across threads that genuinely run at once, with templates for doing so.

* :ref:`Multi-process parallelism via partial loading <partial_loading_parallelism>`: separate processes each
  load, and analyse, part of a large simulation.

* :ref:`Multi-process parallelism via shared memory <using_shared_arrays>`: arrays, and halo catalogues, are
  shared between processes on the same machine without copying.

.. seealso::

   For more general information about performance in pynbody, see :ref:`this page <performance>`.


.. _calculation_parallelism:

Calculation parallelism
-----------------------

The intensive calculations in pynbody run on threads, which do not hold Python's Global Interpreter Lock (GIL),
so they make use of every CPU even on an ordinary build of Python. By default the number of threads is the number
of CPUs on the machine; it is set by the ``number_of_threads`` configuration option, and can be lowered as
described :ref:`below <limiting_threads>`. There are two implementations, which you only need to tell apart if
you build from source:

* *Native threads*, started from Python or in C++. The :class:`~pynbody.kdtree.KDTree` is built and traversed this
  way, and so everything that relies on it is parallel: smoothing lengths and SPH densities, and the SPH operations
  in :mod:`pynbody.sph`. Image rendering (:func:`pynbody.plot.sph.image` and the renderers in
  :mod:`pynbody.sph.renderers`) shares the particles between a thread per CPU.

* *OpenMP* loops in Cython routines: the geometric filters such as :class:`~pynbody.filt.Sphere` and
  :class:`~pynbody.filt.Cuboid`, direct gravity (:func:`pynbody.gravity.direct`), the shrink-sphere centre
  (:func:`~pynbody.analysis.halo.shrink_sphere_center`, which lies behind :func:`pynbody.analysis.center`) and some
  utilities, such as the sums used for virial radii and the search that matches particle IDs between snapshots.
  If you install pynbody from a binary wheel, as most people do, OpenMP is already built in; if you build from
  source, see :ref:`openmp-fix`.

Many routines take a ``num_threads`` argument (for example :meth:`~pynbody.snapshot.simsnap.SimSnap.build_tree`)
to override the configured number for one call. The results do not depend on the number of threads, except in
their last bits: a sum accumulated by several threads (a centre of mass, say, or a quantity summed over particle
pairs) adds its terms in a different order, and so can differ from the serial sum by rounding; and, as noted in the
documentation of :class:`~pynbody.kdtree.KDTree`, a tree built from single-precision positions can place particles
with exactly equal coordinates differently. Pass ``num_threads=1`` if you need bit-for-bit reproducibility.


.. _parallel_io:

Reader parallelism
------------------

RAMSES outputs, and gadget-HDF-derived formats (such as SWIFT, EAGLE, Arepo, GIZMO and Gadget-4 snapshots), are read
in two stages by two kinds of thread: *input* threads read files, and *decode* threads turn what was read -- Fortran
records, or compressed HDF5 chunks -- into arrays. Other formats are read in the calling thread.

By default there is one input thread, since a local disk is read fastest by one reader; but on a parallel
filesystem such as Lustre, GPFS or BeeGFS, where the files of a snapshot usually live on different servers, up
to 16 files are read at once. Decoding uses up to 8 threads, but no more than there are CPUs (and, for HDF5,
only when the data are compressed; otherwise there is nothing to decode).

These numbers are independent of ``number_of_threads``, and are set in the :ref:`configuration <configuration>`:

* in the ``[ramses]`` section, ``parallel-read`` is the most decoding threads (or ``auto`` for one per CPU; a value
  of 1 or less reads everything in the calling thread), and ``io-threads`` the number of input threads (or
  ``auto`` as above);

* in the ``[hdf-bulk-read]`` section, ``decode-threads`` and ``io-threads`` play the same roles, with the limits
  behind ``auto`` in ``max-decode-threads`` and ``parallel-filesystem-io-threads``.

The default configuration file describes these and related options, such as the memory that decoding may use.
The reading threads release the GIL, so even on an ordinary build of Python the input and decoding of a file go
on at the same time as each other, and as any Python code of your own.


.. _limiting_threads:

Limiting parallelism
--------------------

By default pynbody is as responsive as possible, using every CPU. Sometimes that is not what you want: you may
need to leave CPUs for other users or processes on a shared machine; you may be running several pynbody processes
at once, in which case each should take a share; or -- the case treated at length :ref:`below <free_threading>` --
you may be running your own analysis on several threads, in which case pynbody starting a full set of threads for
each of them oversubscribes the CPUs and slows everything down.

Permanently, or for a session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``number_of_threads`` configuration option sets the number of threads that every parallel calculation uses.
During a session, type for example

.. sourcecode:: python

    pynbody.config['number_of_threads'] = 2

To make the change permanent, create a ``.pynbodyrc`` file in your home directory with the following section:

.. code-block:: none

    [general]
    number_of_threads: 2

More information on the pynbody configuration system is :ref:`available here <configuration>`. This option does
not affect the threads that read RAMSES and HDF5 files, which have their own options (see
:ref:`above <parallel_io>`).

For a while: ``thread_limit``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 2.9.0

To limit pynbody's threads only while a block of code runs, use :class:`pynbody.thread_limit
<pynbody.util.threads.thread_limit>`:

.. sourcecode:: python

    with pynbody.thread_limit():
        ...   # each pynbody operation here uses one thread

    with pynbody.thread_limit(2):
        ...   # ... or at most two

Inside the block, each pynbody operation -- building a KD-tree, SPH smoothing, rendering an image, evaluating a
filter, or decoding data read from a file -- uses at most ``n`` threads for computation (``n`` being 1 unless
given). The precise semantics matter when you combine it with threads of your own:

* The limit is process-wide: it applies to every thread while any thread is inside the block. That is what makes it
  apply to a thread pool started inside the block. If blocks with different limits are active at once, the smallest
  applies.

* It limits each operation, not pynbody as a whole: operations running at the same time in different threads may
  each use ``n`` threads. With eight of your threads each analysing a halo under ``thread_limit()``, eight threads
  compute.

* It only ever reduces. The limit is applied on top of ``number_of_threads``, and of any ``num_threads`` passed
  explicitly to a routine; it never raises them.

* It is checked when each operation starts. An operation already running when the block is entered keeps the
  threads it started with, and one started inside the block keeps its threads if the block ends.

* Threads that only read files are *not* limited, since they spend their time waiting for the filesystem rather
  than computing, and on a parallel filesystem reading many files at once is where the speed comes from. Under a
  limit, though, they hand what they read to at most ``n`` decoding threads, instead of decoding it themselves
  (see :func:`~pynbody.util.bulk_read.pipeline.apply_thread_limit`). To limit the reading threads as well, use the
  ``io-threads`` options described :ref:`above <parallel_io>`.

Together with other libraries: threadpoolctl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ is installed, its ``threadpool_limits`` limits pynbody
in the same way as ``thread_limit``, under the API name ``"pynbody"``, so that you can limit pynbody together with
numerical libraries such as BLAS and OpenMP in one place:

.. sourcecode:: python

    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        ...

    with threadpool_limits(1, user_api="pynbody"):
        ...   # limits pynbody alone


.. _free_threading:

Free-threaded analyses
----------------------

.. versionadded:: 2.9.0

In the free-threaded build of Python there is no GIL, so Python code in separate threads really runs at the same
time, and the natural way to analyse many halos at once is a thread pool, without sharing arrays between processes.
Pynbody supports this build from version 2.9.0. The templates below also run correctly on an ordinary build of
Python -- they are executed on one, when this documentation is built -- but there the threads take turns to run
Python code, so you gain only from the parts of the work that pynbody does in compiled code.

Setting up
^^^^^^^^^^

You need the free-threaded build of Python 3.14 or later, which is installed alongside the ordinary one and is
usually called ``python3.14t`` (the installers from python.org offer it as an option; ``uv python install 3.14t``
and conda-forge's ``python-freethreading`` package are alternatives). Pynbody publishes binary wheels for it, so
``pip install pynbody`` works as usual.

Importing an extension module that has not declared itself safe to run without the GIL turns the GIL back on, for
the rest of the process, with a ``RuntimeWarning`` saying which module did it. Pynbody and its core dependencies
(numpy, scipy, h5py, matplotlib) have made the declaration, but at the time of writing some commonly used
packages have not -- for example healpy, which pynbody itself imports only to render healpix images, and parts of
astropy such as ``astropy.table`` (and so ``astropy.cosmology``). To keep the GIL off regardless, run with the
environment variable ``PYTHON_GIL=0`` (or ``python3.14t -X gil=0``). This is at your own risk, in that the module
in question has not been checked for thread safety; the risk is small when, as is usually the case, only one of
your threads uses the module. You can check the state at any time with ``sys._is_gil_enabled()``.

What is safe to do at once
^^^^^^^^^^^^^^^^^^^^^^^^^^

Several threads can safely *read* from the same snapshot at once: pynbody locks the snapshot while loading or
deriving an array, so a thread that needs an array not yet present gets it loaded once, and the others wait for it.
Beyond that, the rules are those of any shared data structure:

* Modifying a shared snapshot, or a shared array, from one thread while others use it is unsafe. Centring or
  rotating (for example with :func:`pynbody.analysis.center` or :func:`~pynbody.analysis.angmom.faceon`) moves the
  *whole* snapshot, not just the halo it was given; converting to physical units rewrites every array; and writing
  to an array changes what the other threads see. Do such things once, before starting the threads -- or give each
  thread a snapshot of its own, as described :ref:`below <copy_on_access_threads>`.

* The execution modes of a snapshot, such as ``immediate_mode`` and ``lazy_off`` (see
  :class:`~pynbody.snapshot.simsnap.SimSnap`), apply only in the thread that enters them. A thread that wants
  ``immediate_mode`` enters it itself, and does not affect the others. (This is also true on ordinary builds of
  Python, from pynbody 2.9.0; previously the modes were shared between threads.)

* Most pynbody routines are themselves parallel (see :ref:`above <calculation_parallelism>`), so running them from
  many threads at once oversubscribes the CPUs. Run the pool inside ``thread_limit``, as the templates do.

Template: a read-only analysis of each halo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest case is an analysis that only reads the snapshot. Here each thread finds the shrink-sphere centre of
a halo, with :func:`~pynbody.analysis.halo.shrink_sphere_center` -- rather than
:func:`~pynbody.analysis.center`, which would move the whole snapshot. Everything that applies to the whole
snapshot (here, loading the halo catalogue) is done once, before the threads start:

.. ipython::

    In [1]: import concurrent.futures

    In [2]: import pynbody

    In [3]: f = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')

    In [4]: f.physical_units()

    In [5]: halos = f.halos()

    In [6]: halos.load_all()

    In [7]: def find_centre(halo):
       ...:     return pynbody.analysis.halo.shrink_sphere_center(halo)
       ...:

    In [8]: targets = [halos[i] for i in range(1, 9)]

    In [9]: with pynbody.thread_limit():
       ...:     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
       ...:         centres = list(pool.map(find_centre, targets))
       ...:

    In [10]: centres[0]

Each of the four pool threads uses one thread for its computation, so four halos are analysed at once on four
CPUs. The results are the same as from a serial loop:

.. ipython::

    In [11]: import numpy as np

    In [12]: all(np.allclose(find_centre(halo), centre) for halo, centre in zip(targets, centres))

.. _copy_on_access_threads:

Template: each halo in its own frame, with copy-on-access snapshots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many analyses need each halo in its own frame -- centred, and perhaps rotated -- which, as noted above, means
modifying the snapshot. To do this in many threads at once, give each thread a snapshot of its own to modify, using
:meth:`~pynbody.snapshot.simsnap.SimSnap.get_copy_on_access_simsnap`. This returns a snapshot containing the same
particles, which copies each array from the original the first time it is used. Transforming it, or changing or
deriving its arrays, then affects only the copy, so the original snapshot stays put and the other threads are
unaffected.

Continuing from the setup above, the following computes a spherically averaged dark matter density profile for
each halo, each in its own frame:

.. ipython::

    In [13]: def density_profile(halo):
       ....:     local = halo.get_copy_on_access_simsnap()
       ....:     pynbody.analysis.center(local)  # moves only this copy
       ....:     profile = pynbody.analysis.Profile(local.dm, ndim=3, rmin='0.5 kpc', rmax='50 kpc', nbins=20)
       ....:     return profile['density']
       ....:

    In [14]: with pynbody.thread_limit():
       ....:     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
       ....:         profiles = list(pool.map(density_profile, targets))
       ....:

    In [15]: profiles[0]

The original snapshot has not moved: halo 1 is still where the halo finder put it, rather than at the origin.

.. ipython::

    In [16]: pynbody.analysis.halo.shrink_sphere_center(halos[1])

Note that:

* The arrays the copy uses are copied, so each thread holds a copy of the data for its own halo. This takes memory
  in proportion to the halos being analysed at once, not to the whole snapshot.

* Do anything that applies to the whole snapshot -- such as converting it to physical units, or loading the halo
  catalogue -- once, before starting the threads, rather than in each thread. Here that is the setup of the
  previous template.

* Analyses like this do much of their work in Python, so on an ordinary build of Python they gain little from
  threads; there, use separate processes, as described next.


.. _partial_loading_parallelism:

Multi-process parallelism via partial loading
---------------------------------------------

On an ordinary build of Python, or across more than one machine, the way to run Python code in parallel is in
separate processes. One approach needs no communication between them at all: each process loads only part of the
simulation, and analyses what it has. This is also the way to analyse a simulation too large for the memory of
any one process.

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


.. _using_shared_arrays:

Multi-process parallelism via shared memory
-------------------------------------------

.. versionadded:: 2.0

  Previously, pynbody had a hidden shared memory system that was used internally and by
  `tangos <https://pynbody.github.io/tangos/>`__ to share arrays between processes. This has
  been exposed for general use in pynbody 2.0.

The other approach to multi-process analysis is for the processes to share the simulation's arrays, through
shared memory, so that it is loaded once and each process can work on any part of it. Pynbody includes the bare
bones of such a framework, for processes on the same machine.

We strongly recommend that you use pynbody's shared memory support with an external framework like
`tangos <https://github.com/pynbody/tangos>`__, which provides a much higher-level interface, and uses it to
analyse large numbers of halos within a single simulation efficiently. However, if you want to develop a
lower-level parallel approach, here is a quick template for how you might do it.

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
`tangos <https://github.com/pynbody/tangos>`__, which hides these details away.

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

.. note::
    Pynbody arrays are just wrappers around numpy arrays, so if you want to write your own parallel processing
    of them -- for example with `Cython <http://cython.org>`_ and OpenMP -- the standard techniques apply. One
    complication is that different OpenMP libraries in one process can conflict; see :ref:`openmp-fix`.

Transferring a halo catalogue to another process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
