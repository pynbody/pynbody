.. halo tutorial


.. _halo_tutorial:

Halos in Pynbody
=======================

Finding the groups of particles that represent galaxies is the key first
step of simulation analysis.  There are several public group finders
available today that find these groups of particles.
``Pynbody`` includes interfaces to several commonly used group finders.

Groups that are virialized are called "halos",
so groups are available in ``pynbody`` using the
:class:`~pynbody.snapshot.SimSnap.halos` function on a
simulation object (``SimSnap``).  When :class:`~pynbody.snapshot.SimSnap.halos`
is called, ``pynbody`` creates
a :class:`~pynbody.halo.HaloCatalogue` that consists of
:class:`~pynbody.halo.Halo` objects.
The ``Halo`` object holds information about the particle IDs and other properties
about a given halo.  A :class:`~pynbody.halo.HaloCatalogue` is an
object is a compilation of all the halos in a given snapshot.

Some possible halo finders that pynbody recognises include:

 - `Amiga Halo Finder (AHF) <http://popia.ft.uam.es/AHF/Download.html>`_ (:class:`~pynbody.halo.AHFCatalogue`);
 -  `Rockstar <https://bitbucket.org/pbehroozi/rockstar-galaxies>`_ (:class:`~pynbody.halo.RockstarCatalogue`);
 - SKID (:class:`~pynbody.halo.GrpCatalogue` class);
 - SubFind (:class:`~pynbody.halo.SubfindCatalogue`).

The :func:`~pynbody.snapshot.SimSnap.halos` function in
:class:`~pynbody.snapshot.SimSnap`
automatically determines whether halo data exists
on disk already for ``pynbody`` to read, or if it should run a halo
finder to create halo data.

This tutorial will show you how to setup and configure pynbody to best use
the group finder functionality built into pynbody. If you are not familiar with
``Pynbody`` in general, it is recommended that you first have a look at
the :ref:`snapshot_manipulation` tutorial.

Configuration
-------------

``Pynbody`` reads a number of different halo formats including the popular
``subfind``. However, it is most comfortable with either AHF or (more
experimentally) Rockstar and can in many cases actually run these codes
for you if you haven't already generated halo catalogues for your simulation.

Rockstar
^^^^^^^^

To install Rockstar, grab the code from Peter Behroozi's bitbucket
repository, make it, and copy it into your ``$PATH``
::

	> git clone https://bitbucket.org/pbehroozi/rockstar-galaxies.git
	> cd rockstar-galaxies; make
	> cp rockstar-galaxies ~/bin/

AHF
^^^

To install AHF, take the most recent version from Alexander Knebe's AHF
page, uncompress it
::

	> wget http://popia.ft.uam.es/AHF/files/ahf-v1.0-084.tgz
	> tar zxf ahf-v1.0-084.tgz; cd ahf-v1.0-084

Edit Makefile.config appropriate for your code, make AHF,
and copy it into your $PATH.
::

	> make AHF
	> cp AHF-v1.0-084 ~/bin/

Now ``pynbody`` will use one of these halo finders to create group files
you can use to analyze your simulations.

Configuration
^^^^^^^^^^^^^

As described in :ref:`configuration`, you can tell pynbody which group
finder you prefer in your configuration file, ``~/.pynbodyrc``.  In the ``general``
section, you can arrange the priority of halo finders to use as you like.


Working with Halos and Catalogues
---------------------------------

We will use the AHF catalogue here since that is the one that is
available for the sample output in the ``testdata`` bundle. The SubFind specific
halo / subhalo structure is handled later.

.. ipython::

 In [1]: import pynbody, matplotlib.pylab as plt

 In [2]: s = pynbody.load('testdata/g15784.lr.01024.gz')

 In [3]: s.physical_units()

We've got the snapshot loaded, now we ask ``pynbody`` to load any
available halo catalogue:

.. ipython::

 In [3]: h = s.halos()

``h`` is the halo catalogue.

.. note:: If the halo finders have to run to find the groups, they may take
   	some time.  AHF typically takes 5 minutes for a million particle
	simulation while Rockstar takes 5-10 minutes running on a single
	processor.

We can easily retrieve some basic
information, like the total number of halos in this catalogue:

.. ipython::

 In [4]: len(h)

To actually access a halo, use square bracket syntax. For example, the following
returns the number of particles in halos 1 and 2

.. ipython::

 In [5]: len(h[1]), len(h[2])

The catalogue has halos ordered by number of particles, so the first
halo for this zoom simulation will be the one we would most likely be
interested in. Halo IDs begin with 1 for many halo finders (including AHF,
which is the sample file being used here).

As may now be evident, "halos" are treated using the
:class:`~pynbody.snapshot.SubSnap` class. The syntax for dealing
with an individual halo therefore precisely mirrors the syntax for
dealing with an entire simulation. For example, we can get the total mass
in halo 1 and see the position of its first few particles as follows:

.. ipython::

 In [10]: h[1]['mass'].sum().in_units('1e12 Msol')

 In [8]: h[1]['pos'][:5]

A really common use-case is that one wants to center the simulation on
a given halo and analyze some of its properties. Since halos are just
:class:`~pynbody.snapshot.SubSnap` objects, this is easy to do:

.. ipython::

 In [1]: pynbody.analysis.halo.center(h[1])

 @savefig halo1_image.png width=5in
 In [2]: im = pynbody.plot.image(h[1].d, width = '500 kpc', cmap=plt.cm.Greys, units = 'Msol kpc^-2')


Halo catalogue information
--------------------------

Any additional information generated by the halo finder
is available through the ``properties`` dictionary associated with halos. For
example

.. ipython::

 In [5]: h[1].properties['children']

returns a list of sub-halos of this halo. Here there are no sub-halos, so
we've been returned an empty list. To see everything that is
known about the halo one can use the standard python dictionary method ``keys``:

.. ipython::

 In [6]: h[1].properties.keys()


Dealing with big simulations and lots of halos
----------------------------------------------

Sometimes, simulations are too large to fit in the memory of your analysis
machine. On the other hand, pynbody never actually loads particle data until
it's needed so it is possible to load a halo catalogue anyway.

Consider the following example.

.. ipython::

 In [2]: f = pynbody.load("testdata/g15784.lr.01024")

 In [3]: h = f.halos()

 In [4]: h[2].properties['mass']/1e12 # another property calculated by AHF in Msol/h

 In [5]: len(h[2])

At no point does this load data from the simulation file; it only accesses the
halo catalogue. In fact, with some formats (including AHF, which is what's
in our sample test data here), you can specify ``dummy=True`` to load only the
properties dictionary:

.. ipython:: :okexcept:

 In [3]: h = f.halos(dummy=True)

 In [4]: h[2].properties['mass'] # this is still OK

 In [5]: len(h[2]) # this, of course, is unknown

.. note::

 The remainder of this section requires the underlying snapshot loader
 to support partial loading, which is currently only the case for *tipsy*
 and *nchilada* formats. See :ref:`loaders`.

Combined with pynbody's partial-loading system, one can go further and
pull only a single halo into your computer's memory at once. The following
example shows you how:

.. ipython::

 In [1]: h2data = h.load_copy(2)

 In [2]: len(h2data) # this is correct again

 In [3]: h2data['mass']

As you can see from the last line, you can now access particle arrays
but the key difference is that ``h2data`` as constructed above only loads the
particles that are required. Conversely
accessing arrays
directly from ``h[2]`` actually loads the full simulation array into memory, even
if only part of it is ever going to be used.




Write halo catalog (i.e. convert AHF outfiles to tipsy format)
--------------------------------------------------------------

Tipsy is a particle viewer.  A tipsy format file can be useful for
quick viewing in tipsy to check whether the AHF halo finder did
anything sensible. Write the (ahf) halo catalog to disk. Former idl
users might notice that this produces outfiles similar to 'Alyson's
idl script'.

The 3 written file types are:

1.   .gtp (tipsy file with halos as star particles);
2.   .grp (ascii halo id of every snapshot particle, 0 if none);
3.   .stat (ascii condensed version of AHF halos file).

This halo file set emulates the halo finder SKID. Tipsy and skid can be found at
`<http://www-hpcc.astro.washington.edu/tools/>`_.


Working with SubFind Halos and Subhalos
---------------------------------------

If using the Gadget3 SubFind HDF5 output (for example, OWLS / Eagle or Smaug sims)
most of the examples from AHF above can be used, except for the subhalos structure.
One major change is that the halo catalogue is a separate file to the snapshot.

.. ipython::

 In [1]: import pynbody, matplotlib.pylab as plt

 In [2]: s = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/snapshot_103/snap_103.hdf5')

 In [3]: s.physical_units()

We've got the snapshot loaded and can access the particle data in any manner we
like as usual but unlike AHF we can't load halos. Instead to get ``pynbody``
to load the halo catalogue we have to access the subfind output directly:

.. ipython::

 In [3]: s = pynbody.load('testdata/Test_NOSN_NOZCOOL_L010N0128/data/subhalos_103/subhalo_103')

 In [2]: s.physical_units()

 In [4]: h = s.halos()

``h`` is the Friends-of-Friends (FOF) halo catalogue, upon which SubFind is based.

As with the AHF example we can easily retrieve some basic
information, like the total number of halos in this catalogue:

.. ipython::

 In [5]: len(h), h.ngroups, h.nsubhalos

Where the last value is the number of subhalos, see next section on these.
To actually access a halo, use square bracket syntax as before.
For example, the following returns the number of particles in halos 1 and 2

.. ipython::

 In [6]: len(h[1]), len(h[2])

The catalogue has FOF halos ordered by number of particles, so the first
halo for this small box simulation will be the largest object.
Halo IDs begin with 0 for SubFind / FOF unlike AHF.

The "halos" are treated using the
:class:`~pynbody.gadgethdf.SubFindHDFSnap` class. The syntax for dealing
with an individual halo is the same as AHF and the snapshot simulation.
For example, we can get the total mass in the second FOF halo
and see the position of its first few particles as follows:

.. ipython::

 In [7]: h[1]['mass'].sum().in_units('1e12 Msol')

 In [8]: h[1]['pos'][:5]

A really common use-case is that one wants to center the simulation on
a given halo and analyze some of its properties:

.. ipython::

 In [9]: pynbody.analysis.halo.center(h[1], vel=False)

 @savefig halo1_image_subfind.png width=5in
 In [10]: im = pynbody.plot.image(h[1].d, width = '40 kpc', cmap=plt.cm.Greys, units = 'Msol kpc^-2')


Subhalo catalogue information
-----------------------------

After the FOF group has been found, SubFind runs on this reduced particle list
to determine gravitational bound substructures (or subhalos) within the larger FOF halo.
To access the list of subhalos simply call:

.. ipython::

 In [11]: h[1].sub[:]

to return a list of sub-halos of this halo. Then one can select subhalo particles as
before (e.g. dark matter velocities):

.. ipython::

 In [12]: h[1].sub[0].d['vel']

for the main (i.e. first) subhalo of the second FOF halo. As with AHF additional halo
catalogue values such as the centre of mass, or the velocity dispersion, can be accessed
by the properties list for each halo / subhalo. Note that the subhalo properties list
is far more extensive than the FOF halo:

.. ipython::

 In [13]: h[2].properties

 In [14]: h[2].properties['CenterOfMass']

 In [15]: h[2].sub[4].properties

 In [16]: h[2].sub[4].properties['CenterOfMass']

To access the entire dataset of a given property (say all of the Stellar
Velocity Dispersions) requires an embedded for loop over the HDF5 catalogue and
appending to an array:

.. ipython::

 In [17]: SubStellarVelDisp = [[subhalo.properties['SubStellarVelDisp'] for subhalo in halo.sub] for halo in h]

 In [19]: SubStellarVelDisp[5]
