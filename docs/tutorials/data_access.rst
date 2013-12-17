.. data_access tutorial

.. _data-access:

A walk through pynbody's low-level facilities
=============================================

The following example shows how to load a file, determine various
attributes, access some data and make use of unit information. 

If you're more interested in making pretty pictures and plots straight
away, you may wish to read the :ref:`basic facilities tutorial
snapshot_manipulation` first.

.. note:: This tutorial discusses an
  interactive session rather than a script. This is only cosmetically
  different; the commands discussed here can of course all be used in
  a script too. We will use the `ipython` interpreter which offers a
  much richer interactive environment over the vanilla `python`
  interpreter. However, you can type exactly the same commands into
  vanilla `python`; only the formatting will look slightly
  different. For instance, the `ipython` prompt looks like ``In [1]:``
  while the `python` prompt looks like ``>>>``.


First steps
-----------

.. note:: Before you start make sure `pynbody` is properly
 installed. See :ref:`pynbody-installation`
 for more information. You will also need the standard `pynbody` test
 files, so that you can load the exact same data as used to write the
 tutorial. You need to download these separately here: <https://code.google.com/p/pynbody/downloads/list>
 (`testdata.tar.gz`).

After you have extracted the testdata folder (e.g. with ``tar -xzf
testdata.tar.gz``), launch `ipython`. At the prompt, type ``import
pynbody``. If all is installed correctly, this should silently
succeed, and you are ready to use `pynbody` commands. Here's an
example. We'll also load the `numpy` module as it provides some
functions we'll make use of later.

.. ipython::

 In [1]: import pynbody

 In [1]: import numpy as np 

 In [2]: f = pynbody.load("testdata/test_g2_snap")
 Attempting to load as <class 'pynbody.gadget.GadgetSnap'>

Here we've loaded a sample gadget file. Not much seems to have
happened when you called :func:`pynbody.load`, but the variable ``f``
is now your calling point for accessing data.

In fact ``f`` is an object known as a `SimSnap` (see
:class:`pynbody.snapshot.SimSnap`).

Behind the scenes, the function inspects the provided path and decides
what file format it is. At the time of writing, supported file formats
include tipsy, nchilada, gadget-1, gadget-2, gadget-HDF and
ramses. For most purposes you should never need to know what type of
file you are dealing with -- that's the whole point of the `pynbody`
framework.

.. note:: If you look inside the `testdata` folder, you'll notice that
 our test snapshot is actually an example of a `spanned` gadget
 snapshot. There is not actually a file called ``test_g2_snap``, but
 rather two files from two CPUs, ``test_g2_snap.0`` and
 ``test_g2_snap.1``. `pynbody` knows to load both of these if you ask
 for `test_g2_snap`; if you ask for `test_g2_snap.1` (for instance),
 it'll load only that particular file.

The `SimSnap` object that's returned is currently a fairly empty
holder for data. The data will be loaded from disk as and when you
request it.

Finding out something about the file
------------------------------------

Let's start to inspect the file we've opened. The standard python operator ``len`` can be used to query the number
of particles in the file:

.. ipython::

 In [3]: len(f)
 Out[3]: 8192

We can also find out about particles of a particular type or `family`
as it is known within pynbody. To find out which families are present
in the file, use :func:`~pynbody.snapshot.SimSnap.families`:

.. ipython::

 In [3]: f.families() 
 Out[3]: [<Family gas>, <Family dm>, <Family star>]

You can pick out just the particles belonging to a family by using the
syntax ``f.family``. So, for example, we can see how many particles of
each type are present:

.. ipython::

 In [4]: len(f.dm)
 Out[4]: 4096

 In [5]: len(f.gas)
 Out[5]: 4039

 In [6]: len(f.star)
 Out[6]: 57

Useful information about the file is stored in a python dictionary
called `properties`:

.. ipython::

 In [4]: f.properties

Like any python dictionary, specific properties can be accessed by
name:

.. ipython::

 In[4]: f.properties['a']

These names are standardized across different file formats. Here for example `z`
means redshift, `a` means the cosmological scalefactor, `h` indicates
the Hubble constant in standard units (100 km/s/Mpc). 

.. note:: Actually ``f.properties`` has some behaviour which is
 very slightly different from a normal python dictionary. For further
 information see :class:`~pynbody.simdict.SimDict`.


Retrieving data
---------------

Like ``f.properties``, ``f`` itself also behaves like a python
dictionary. The standard python method
``f.``:func:`~pynbody.snapshot.SimSnap.keys` returns a list of arrays
that are currently in memory.

.. ipython::

  In [7]: f.keys()
  Out[7]: ['eps']

Right now it's empty! That's actually correct because data is only
retrieved when you first access it. To find out what `could` be loaded,
use the `pynbody`-specific method
``f.``:func:`~pynbody.snapshot.SimSnap.loadable_keys`.

.. ipython::

  In [10]: f.loadable_keys()
  Out[10]: ['pos', 'vel', 'id', 'mass']

This looks a bit more promising.
To access data, simply use the normal dictionary syntax. For example
``f['pos']`` returns an array containing the 3D-coordinates of all the
particles. 

.. ipython::

 In [11]: f['pos']
 Out[11]: 
 SimArray([[   53.31897354,   177.84364319,   128.22311401],
       [  306.75045776,   140.44454956,   215.37149048],
       [  310.99908447,    64.1344986 ,   210.53594971],
       ..., 
       [ 2870.90161133,  2940.17114258,  1978.79492188],
       [ 2872.41137695,  2939.21972656,  1983.91601562],
       [ 2863.65112305,  2938.05444336,  1980.06152344]], dtype=float32, 'kpc h**-1')



.. note::

 Array names are standardized across all file
 formats. For instance, even if you load a Gadget-HDF file -- which
 internally refers to the position array as `coordinates` -- you
 still access that array from pynbody by the name ``pos``. The
 intention is that code never needs to be adapted simply because you
 have switched file format. However the name mapping is fully
 :ref:`configurable <configuration>` should you wish to adopt
 different conventions.

Some arrays are stored only for certain families. For example,
densities are stored only for gas particles and are accessed as
``f.gas['rho']``.  To find out what arrays are available for the gas
family, use
``f.gas.``:func:`~pynbody.snapshot.SimSnap.loadable_keys`:

.. ipython::

 In [13]: f.gas.loadable_keys()
 Out[13]: 
 ['nhp',
 'smooth',
 'nhe',
 'u',
 'sfr',
 'pos',
 'vel',
 'id',
 'mass',
 'nh',
 'rho',
 'nheq',
 'nhep']

So, we can get the density of the gas particles like this:

.. ipython::

  In [14]: f.gas['rho']
  Out[14]: 
  SimArray([  1.38886092e-09,   3.36176842e-09,   4.52736737e-09, ...,
         8.53409521e-09,   7.41017736e-09,   1.40517520e-09], dtype=float32, '1.00e+10 h**2 Msol kpc**-3')


.. note:: The :class:`~pynbody.array.SimArray` objects are actually
 `numpy` arrays with some added functionality (such as unit tracking,
 discussed below). Numerical operations are very nearly as fast as
 their numpy equivalents. However, if you want to squeeze the
 performance of your code, you can always get a vanilla numpy array by
 using the `numpy` view mechanism,
 e.g. ``f.gas['rho'].view(type=numpy.ndarray)``

.. _create_arrays :

Creating your own arrays
------------------------

You can create arrays using the obvious assignment syntax:

.. ipython::

  In [14]: f['twicethemass'] = f['mass']*2

You can also define new arrays for one family of particles:

.. ipython::

  In [14]: f.gas['myarray'] = f.gas['rho']**2

An array created in this way exists *only* for the gas
particles; trying to access it for other particles raises an
exception.

Alternatively, you can define *derived arrays* which are calculated (and
re-calculated) on demand. For example,

.. ipython::

  In [3]: @pynbody.derived_array
     ...:def thricethemass(sim) :
     ...:    return sim['mass']*3
     ...: 


At this point, nothing has been calculated. However, when you ask for
the array, the values are calculated and stored

.. ipython::

  In [4]: f['thricethemass']
  
  Out[4]: 
  SimArray([ 1.28755365,  1.28755365,  1.28755365, ...,  1.28755365,
          1.28755365,  1.28755365], '1.00e+10 Msol')

This has the advantage that your new `thricethemass` array is
automatically updated when you change the `mass` array:

.. ipython::
  
  In [4]: f['mass'][0] = 1
  
  In [6]: f['thricethemass']
  SimSnap: deriving array thricethemass
  Out[6]: 
  SimArray([ 3.        ,  1.28755365,  1.28755365, ...,  1.28755365,
          1.28755365,  1.28755365], '1.00e+10 Msol')

Note, however, that the array is not re-calculated every time you
access it, only if the `mass` array has changed. Therefore you don't
waste any time by using derived arrays. For more information see
the reference documentation for :ref:`derived arrays <derived>`.

Keeping on top of units
-----------------------


You might have noticed in the output from the above experiments that
`pynbody` keeps track of unit information whenever it can.

.. warning:: It's worth understanding exactly where pynbody gets this
 information from, in case anything goes wrong. In the case
 of `Ramses`, and `Gadget-HDF` files the unit information is stored
 within your snapshot, and pynbody takes advantage of this. For
 old-style `Gadget` snapshots, the default cosmological gadget setup is
 assumed. For `nchilada` and `tipsy`, an nchilada or gasoline
 ``.param`` file is sought in the directory from which you are loading
 the snapshot and its immediate parent. 

You can print out the units of any given array by accessing the
``units`` property:

.. ipython::

 In [16]: f['mass'].units
 Out[16]: Unit("kpc h**-1")

However, it's usually more helpful to simply convert your arrays into
something more managable than the internal units. `Pynbody` arrays can
be converted using the :func:`~pynbody.array.SimArray.in_units`
function; just pass in a string representing the units you want.

.. ipython::

 In [17]: f['pos'].in_units('Mpc')
 Out[17]: 
 SimArray([[ 0.07509714,  0.25048399,  0.18059593],
       [ 0.4320429 ,  0.19780922,  0.30334011],
       [ 0.43802688,  0.09033027,  0.2965295 ],
       ..., 
       [ 4.04352331,  4.1410861 ,  2.78703499],
       [ 4.04564953,  4.13974571,  2.79424787],
       [ 4.03331137,  4.13810492,  2.78881884]], dtype=float32, 'Mpc')


.. note:: The function :func:`~pynbody.array.SimArray.in_units` returns a copy of
 your array in new units. Next time you access ``f['pos']`` it will be
 back in its original units. If you want to permanently convert the array in-place
 use :func:`~pynbody.array.SimArray.convert_units` or see below.

Another option is to request that `pynbody` converts all your arrays
into something sensible, using
:func:`~pynbody.array.SimSnap.physical_units`,

.. ipython::

 In [18]: f.physical_units()

Take a look at what's happened to the density:

.. ipython::
 
 In [19]: f.gas['rho']
 Out[19]: 
 SimArray([  7.00124788,  16.94667435,  22.82245827, ...,  43.0203743 ,
        37.354702  ,   7.08348799], dtype=float32, 'Msol kpc**-3')

Note that the conversion will also be made when loading any arrays in
future; for example:

.. ipython::

 In [21]: f['vel']
 vel km a**1/2 s**-1 -> km s**-1
 Out[21]: 
 SimArray([[ 27.93829346,   4.98370504, -10.00886631],
       [ 15.36156368,   5.7859726 ,   4.36315632],
       [ -8.35731888,  -2.88852572,  22.8099041 ],
       ..., 
       [ 27.74917603,  85.60175323,  15.53243732],
       [ 40.75585556,  59.44286728,  44.24484634],
       [ 38.38396454,  68.63973236,  46.01428986]], dtype=float32, 'km s**-1')

A new array generated from a unary or binary operation will inherit
the correct units. For example

.. ipython::

 In [55]: 5*f['vel']
 Out[55]: 
 SimArray([[ 139.69146729,   24.9185257 ,  -50.0443306 ],
       [  76.80781555,   28.92986298,   21.81578064],
       [ -41.78659439,  -14.44262886,  114.0495224 ],
       ..., 
       [ 138.74588013,  428.00875854,   77.66218567],
       [ 203.77928162,  297.21432495,  221.22422791],
       [ 191.91983032,  343.19866943,  230.07144165]], dtype=float32, 'km s**-1')

 In [56]: (f['vel']**2).units 
 Out[56]: 
 SimArray([[  780.54821777,    24.83731651,   100.17740631],
       [  235.97764587,    33.47747803,    19.03713226],
       [   69.84477997,     8.3435812 ,   520.29174805],
       ..., 
       [  770.01678467,  7327.66015625,   241.25660706],
       [ 1661.03979492,  3533.45458984,  1957.60644531],
       [ 1473.32873535,  4711.41308594,  2117.31494141]], dtype=float32, 'km**2 s**-2')

 
 In [57]: np.sqrt(((f['vel']**2).sum(axis=1)*f['mass'])).units
 Out[57]: 

You can even associate arrays with the loaded
:class:`~pynbody.snapshot.SimSnap` unit system even when you create
them *outside* the :class:`~pynbody.snapshot.SimSnap`. This is useful
for keeping things tidy with your unit conversions if you are
calculating quantities that don't apply to all of the particles. For
instance:

.. ipython::

 In [6]: array = pynbody.array.SimArray(np.random.rand(10)) # make the newly-formed numpy array a pynbody array

 In [7]: array.sim = f # this links the array to the simulation
 
 In [8]: array.units = 'Mpc a' # we set units that require cosmology information

 In [9]: array

 In [9]: array.in_units('kpc')

Note that the units were correctly converted into physical units in
the last step.

For more information see the reference documentation for
:class:`pynbody.units`.

.. _subsnaps:

Subsnaps
--------

An important concept within `pynbody` is that of a subsnap. These are
objects that look just like a :class:`~pynbody.snapshot.SimSnap` but actually only point
at a subset of the particles within a `parent`. Subsnaps are always
instances of the :class:`~pynbody.snapshot.SubSnap` class.

You've already seen some examples of subsnaps, actually. When you
accessed ``f.gas`` or ``f.dm``, you're given back a subsnap pointing
at only those particles. However, subsnaps can be used in a much more
general way. For example, you can use python's normal array slicing
operations. Here we take every tenth particle:

.. ipython::

 In [24]: every_tenth = f[::10]

 In [25]: len(every_tenth)
 Out[25]: 820

In common with python's normal mode of working, this does not copy any
data, it merely creates another pointer into the existing data. As an
example, let's modify the position of one of our particles in the
new view:

.. ipython::

  In [30]: every_tenth['pos'][1]
  Out[30]: SimArray([ 505.03970337,  439.98474121,  272.89904785], dtype=float32, 'kpc')

  In [27]: every_tenth['pos'][1] = [1,2,3]

  In [28]: every_tenth['pos'][1]
  Out[28]: SimArray([ 1.,  2.,  3.], dtype=float32, 'kpc')

This change is reflected in the main snapshot.

.. ipython::

  In [33]: f['pos'][10]
  Out[33]: SimArray([ 1.,  2.,  3.], dtype=float32, 'kpc')

.. note:: If you're used to numpy's flexible indexing abilities, you
 might like to note that, typically, ``f[array_name][index] ==
 f[index][array_name]``. The difference is that applying the index to
 the whole snapshot is more flexible and can lead to simpler code. In
 particular, ``numpy_array[index]`` may involve copying data whereas
 ``f[index]`` never does; it always returns a new object pointing back at
 the old one.

You can pass in an array of boolean values representing
whether each successive particle should be included (`True`) or not
(`False`).  This allows the use of `numpy`'s comparison
operators. For example:

.. ipython::

 In [40]: f_slab = f[(f['x']>1000)&(f['x']<2000)]
 Out[40]: None
 
 In [41]: f_slab['x'].min()
 Out[41]: SimArray(1000.4244995117188, dtype=float32)
 
 In [42]: f_slab['x'].max()
 Out[42]: SimArray(1999.713134765625, dtype=float32)
 
 In [43]: f['x'].min()
 Out[43]: SimArray(0.16215670108795166, dtype=float32)
 
 In [44]: f['x'].max()
 Out[44]: SimArray(4225.29345703125, dtype=float32)


Here `f_slab` is pointing at only those particles which have
x-coordinates between 1000 and 2000.

Note that subsnaps really do behave exactly like snapshots. So, for
instance, you can pick out sub-subsnaps or sub-sub-subsnaps. 

.. ipython::

 In [45]: len(f_slab.dm)
 
 In [46]: len(f_slab.dm[::10])
 
 In [48]: f_slab[[100,105,252]].gas['pos']

.. note:: Under most circumstances there is very little performance
 penalty to using a `SubSnap`. However in performance-critical code it
 is worth understanding a little more about what's going on under the
 hood. See :ref:`performance`.

Filters
-----------

Another way you can select a subset of particles is to use a
`filter`. This can lead to more readable code than the equivalent
explicitly written condition. For example, to pick out a sphere
centered on the origin, you can use:

.. ipython::

 In [71]: from pynbody.filt import *

 In [72]: f_sphere = f[Sphere('10 kpc')]


For a list of filters, see  :py:mod:`pynbody.filt`.


Where next?
-----------

This concludes the tutorial for basic use of `pynbody`. Further
:ref:`tutorials <tutorials>` for specific tasks are available. We are
happy to provide further assistance via our
`user group email list
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_. 
