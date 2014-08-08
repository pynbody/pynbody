.. profile tutorial


Profiles in Pynbody
===================


Radial Density Profile 
----------------------

Making profiles of all kinds of quantities is easy -- here's a simple
example that shows how to plot a density profile:

.. plot:: tutorials/example_code/density_profile.py
   :include-source:

Below is a more extended description of the
:mod:`~pynbody.analysis.profile` module.

 
The Profile Class 
-----------------

The :class:`~pynbody.analysis.profile.Profile` class is meant to be a
general-purpose class to satisfy (almost) all simulation profiling
needs. Profiles are the most elementary ways to begin analyzing a
simulation, so the :class:`~pynbody.analysis.profile.Profile` class is
designed to be an extension of the syntax implemented in the
:class:`~pynbody.snapshot.SimSnap` class and its derivatives.

Creating a :class:`~pynbody.analysis.profile.Profile` instance simply
defines the bins etc. Importantly, it also stores lists of particle
indices corresponding to each bin, so you can easily identify where
the particles belong.

Let's quickly load a snapshot: 

.. ipython::

  In [1]: import pynbody; from pynbody.analysis import profile; import matplotlib.pylab as plt

  In [1]: s = pynbody.load('testdata/g15784.lr.01024'); s.physical_units()

  In [2]: h = s.halos()

  In [3]: pynbody.analysis.angmom.faceon(h[1])

We're centered on the main halo (as in the cookbook example above) and
we make the :class:`~pynbody.analysis.profile.Profile` instance: 

.. ipython::
  
  In [3]: p = profile.Profile(h[1].s,min='.01 kpc', max='50 kpc')

With the default parameters, the profile is made in the xy-plane. To
make a spherically-symmetric 3D profile, specify ``ndim=3`` when
creating the profile. 

.. ipython::

  In [3]: pdm_sph = profile.Profile(s.d,min='.01 kpc', max = '250 kpc')

Even though we use ``s.d`` here (i.e. the full snapshot, not
just halo 1), the whole snapshot is still centered on halo 1. 

.. note:: You can pass unit strings to ``min`` and ``max`` and the
 conversion will be done automatically into whatever the current
 units of the snapshot are so you don't have to explicitly do any unit conversions.    



Automatically-generated profiles
--------------------------------

Many profiling functions are already implemented -- see the
:class:`~pynbody.analysis.profile.Profile` documentation for a full
list with brief descriptions. You can also check the available
profiles in your session using
:func:`~pynbody.analysis.profile.Profile.derivable_keys` just like you
would for a :class:`~pynbody.snapshot.SimSnap`:

.. ipython::

  In [3]: p.derivable_keys()

Additionally, *any* existing array can be 'profiled'. For example, if
the metallicity [Fe/H] is a derived field stored under 'feh', then
plotting a metallicity profile is as simple as:

.. ipython::

    In [4]: plt.plot(p['rbins'].in_units('kpc'),p['feh'],'k')

    @savefig profile_fig1.png width=5in
    In [5]: plt.xlabel('$R$ [kpc]'); plt.ylabel('[Fe/H]')

If the array doesn't exist but is deriveable (check with
``s.derivable_keys()``), it is automatically calculated.

In addition, you can define your own profiling functions in your code
by using the ``Profile.profile_property`` decorator::

   @profile.Profile.profile_property
   def random(self):
      import numpy as np
      return np.random.rand(self.nbins)

Now this will be automatically derivable for any newly-created profile as ``'random'``.

Calculating Derivatives and Dispersions
---------------------------------------

You can calculate derivatives of profiles automatically. For instance,
you might be interested in d phi / dr if you're looking at a
disk. This is as easy as attaching a ``d_`` to the profile name. For
example:

.. ipython::

   In [6]: p_all = profile.Profile(s,min='.01 kpc', max='250 kpc')

   In [6]: p_all['pot'][0:10] # returns the potential profile

   In [7]: p_all['d_pot'][0:10] # returns d phi / dr from p["phi"]

Similarly straightforward is the calculation of dispersions and
root-mean-square values. You simply need to attach a ``_disp`` or
``_rms`` as a suffix to the profile name. To get the stellar velocity
dispersion:

.. ipython::

    In [7]: plt.plot(p['rbins'].in_units('kpc'),p['vr_disp'].in_units('km s^-1'),'k',hold=False)

    @savefig profile_fig2.png width=5in    
    In [6]: plt.xlabel('$R$ [kpc]'); plt.ylabel('$\sigma_{r}$')


In addition to doing this by hand, you can make a
:class:`~pynbody.analysis.profile.QuantileProfile` that can return any
desired quantile range. By default, this is the mean +/- 1-sigma: 

.. ipython::

    In [5]: p_quant = profile.QuantileProfile(h[1].s, min = '0.1 kpc', max = '50 kpc')

    In [6]: plt.plot(p_quant['rbins'], p_quant['feh'][:,1], 'k', hold=False)

    In [6]: plt.fill_between(p_quant['rbins'], p_quant['feh'][:,0], p_quant['feh'][:,2], color = 'Grey', alpha=0.5)
    
    @savefig profile_quant.png width=5in
    In [6]: plt.xlabel('$R$ [kpc]'); plt.ylabel('[Fe/H]')



Making a profile using a different quantity
-------------------------------------------

Radial profiles are nice, but sometimes we want a "profile" using a
different quantity. We might want to know, for example, how the mean
metallicity varies as a function of age in the
stars. :class:`~pynbody.analysis.profile.Profile` calls the function
:func:`~pynbody.analysis.profile.Profile._calculate_x` by default and
this simply returns the 3D or xy-plane radial distance, depending on
the value of ``ndim``. We can specify a different function using the
``calc_x`` keyword. Often these are simple so a lambda function can be
used (e.g. if we just want to return an array) or can also be more
complicated functions. For example, to make the profile of stars in
halo 1 according to their age:

.. ipython::

   In [6]: s.s['age'].convert_units('Gyr')

   In [5]: p_age = profile.Profile(h[1].s, calc_x = lambda x: x.s['age'], max = '10 Gyr')

   In [6]: plt.plot(p_age['rbins'], p_age['feh'], 'k', label = 'mean [Fe/H]',hold=False)
   
   In [6]: plt.plot(p_age['rbins'], p_age['feh_disp'], 'k--', label = 'dispersion') 
   
   In [6]: plt.xlabel('Age [Gyr]'); plt.ylabel('[Fe/H]')

   @savefig profile_fig4.png width=5in
   In [6]: plt.legend()


Vertical Profiles and Inclined Profiles
---------------------------------------

For analyzing disk structure, it is frequently useful to have a
profile in the z-direction. This is done with the
:class:`~pynbody.analysis.profile.VerticalProfile` which behaves in
the same way as the :class:`~pynbody.analysis.profile.Profile`. Unlike
in the basic class, you must specify the radial range and maximum z to
be used:

.. ipython::

   In [5]: p_vert = profile.VerticalProfile(h[1].s, '3 kpc', '5 kpc', '5 kpc')

   In [5]: plt.plot(p_vert['rbins'].in_units('pc'), p_vert['density'].in_units('Msol pc^-3'),'k', hold=False)

   @savefig profile_fig5.png width=5in
   In [5]: plt.xlabel('$z$ [pc]'); plt.ylabel(r'$\rho_{\star}$ [M$_{\odot}$ pc$^{-3}$]')


Similarly, one can make inclined profiles using the
:class:`~pynbody.analysis.profile.VerticalProfile`, but the snapshot needs to be rotated first: 

.. ipython::

   In [5]: s.rotate_x(60) # rotate the snapshot by 60-degrees

   In [5]: p_inc = profile.InclinedProfile(h[1].s, 60, min = '0.1 kpc', max = '50 kpc')


