.. data_access tutorial


Getting started: basic data access
====================================

The following example shows how to load a file, determine various
attributes, access some data and make use of unit information. 

.. note:: Unlike other `pynbody` tutorials, here we will discuss an
  interactive session rather than a script. This is only cosmetically
  different; the commands discussed here can of course all be used in
  a script too. We will use the `ipython` interpreter which offers a
  much richer interactive environment over the vanilla `python`
  interpreter. However, you can type exactly the same commands into
  vanilla `python`; only the formatting will look slightly
  different. For instance, the `ipython` prompt looks like``In [1]:``
  while the `python` prompt looks like ``>>>``.


First steps
--------------

.. note:: Before you start make sure `pynbody` is properly
 installed. See <https://code.google.com/p/pynbody/wiki/Installation>
 for more information. You will also need the standard `pynbody` test
 files, so that you can load the exact same data as used to write the
 tutorial. These files are in the `nose` folder alongside the unit tests. You already have these if you used `mercurial` to fetch the
 latest version; otherwise you'll need to download them separately
 here: <https://code.google.com/p/pynbody/downloads/list>
 (`nose.tar.gz`).

Change into the `nose` folder (see the note above if you can't find
this) so the test files are easy to access. Then launch `ipython`. At
the prompt, type ``import pynbody``. If all is installed correctly,
this should silently succeed, and you are ready to use `pynbody`
commands. Here's an example.

.. ipython::

 In [1]: import pynbody


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
------------------------------

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
------------------------------------------

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


.. note:: Array names are standardized across all file
 formats. For instance, even if you load a Gadget-HDF file -- which
 internally refers to the position array as `coordinates` -- you
 still access that array from pynbody by the name ``pos``. The
 intention is that code never needs to be adapted simply because you
 have switched file format. However the name mapping is fully
 configurable (see <configuration>) should you wish to adopt
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



Keeping on top of units
-----------------------------------------------------


You might have noticed in the output from the above experiments that
`pynbody` keeps track of unit information whenever it can.

.. warning:: It's worth understanding exactly where pynbody gets this
 information from, so you can anticipate when it might be wrong.  In the case
 of `Ramses`, and `Gadget-HDF` files the unit information is stored
 within your snapshot, and pynbody takes advantage of this. For
 old-style `Gadget` snapshots, the default cosmological gadget setup is
 assumed. For `nchilada` and `tipsy`, an nchilada or gasoline
 ``.param`` file is sought in the directory from which you are loading
 the snapshot and its immediate parent. 

At the simplest level you can simply print out the units of any given
array by accessing the ``units`` property:

.. ipython::

 In [16]: f['pos'].units
 Out[16]: Unit("kpc h**-1")

However, it's usually more helpful to simply convert your arrays into
something more managable than the internal units. `Pynbody` arrays can
be converted using the :func:`~pynbody.array.SimArray.in_units`
function; simply pass in a string representing the units you want.

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



Conclusion
-----------

We've scratched the surface of the capabilities of `pynbody`. 

..note:: We should insert pointers to other tutorials here

