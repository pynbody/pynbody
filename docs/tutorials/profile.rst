.. _profile:


Profiles
========

The :class:`~pynbody.analysis.profile.Profile` class is a
general-purpose class that can be used to bin information in a simulation
in various ways. The :class:`~pynbody.analysis.profile.Profile` class is
designed to be an extension of the syntax implemented in the
:class:`~pynbody.snapshot.SimSnap` class.

Creating a :class:`~pynbody.analysis.profile.Profile` instance
defines the bins, after which specific calculations can be carried out.

To illustrate, let's load a snapshot:

.. ipython::

  In [1]: import pynbody;
     ...: from pynbody.analysis import profile;
     ...: import matplotlib.pylab as plt

  In [1]: s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024')

  In [1]: s.physical_units()

  In [2]: h = s.halos()

  In [3]: pynbody.analysis.faceon(h[0])

The final command here centres the origin on the main halo and puts the disk in the xy-plane
(see :ref:`aligning`). Now we can make the :class:`~pynbody.analysis.profile.Profile` instance:

.. ipython::

  @suppress
  In [4]: plt.clf()

  In [3]: p = profile.Profile( h[0].star, rmin = '.05 kpc', rmax = '50 kpc')

.. note:: You can pass either floating point values or unit strings to ``rmin`` and ``rmax``.

With the default parameters, a 2D profile is created, in the xy-plane. Our use of the
:func:`~pynbody.analysis.faceon` function ensures that the stellar disk is ready
for analysis in this way. We can now plot the density profile:

.. ipython::

  In [4]: plt.plot(p['rbins'].in_units('kpc'),p['density'].in_units('Msol kpc^-2'),'k')

  @savefig profile_stellar_den.png width=5in
  In [5]: plt.xlabel('$R$ [kpc]')
     ...: plt.ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
     ...: plt.semilogy()


The binning is linear by default, but we can see here that the profile is not well-sampled
at large radii. We can change the binning to logarithmic by specifying ``type='log'``:

.. ipython:: python

  @suppress
  plt.clf()

  p = profile.Profile( h[0].star, rmin = '.05 kpc', rmax = '50 kpc', type='log')

  plt.plot(p['rbins'].in_units('kpc'),p['density'].in_units('Msol kpc^-2'),'k')

  @savefig profile_stellar_den_logbin.png width=5in
  plt.xlabel('$R$ [kpc]'); \
  plt.ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]'); \
  plt.semilogy()



To make a spherically-symmetric 3D profile, specify ``ndim=3`` when
creating the profile.

.. ipython::

  In [3]: pdm_3d = profile.Profile(s.dm, rmin = '.01 kpc', rmax = '500 kpc', ndim = 3)

Even though we use ``s.dm`` here (i.e. dark matter from the full snapshot, not
just halo 0), the whole snapshot is still centered on halo 0 following our earlier call to
:func:`~pynbody.analysis.faceon`. This allows us to explore
that far outer reaches of the halo around the galaxy. Let's now plot the dark matter
density profile:

.. ipython::

  @suppress
  In [4]: plt.clf()

  In [4]: plt.plot(pdm_3d['rbins'].in_units('kpc'),pdm_3d['density'].in_units('Msol kpc^-3'),'k')

  @savefig profile_dm_den.png width=5in
  In [5]: plt.xlabel('$r$ [kpc]'); plt.ylabel(r'$\rho_{\rm DM}$ [M$_{\odot}$ kpc$^{-3}$]'); plt.loglog()





Mass-weighted average quantities
--------------------------------

The above examples illustrate the most basic use of profiling, to generate binned density
estimates. One may also generate mass-weighted averages of *any* quantity that is either
stored in the snapshot or derivable from it. For example, the sample snapshot being used
above has metallicity information from which an Fe/H estimate can be derived by pynbody.

.. ipython::

    @suppress
    In [4]: plt.clf()

    In [4]: plt.plot(p['rbins'].in_units('kpc'),p['feh'],'k')

    @savefig profile_fig1.png width=5in
    In [5]: plt.xlabel('$R$ [kpc]'); plt.ylabel('[Fe/H]')

Special quantities
------------------

As well as straight-forward densities and mass-weighted averages, there are a number of
special profiling functions implemented. To see a full list, use the
:meth:`pynbody.analysis.profile.Profile.derivable_keys` method or consult
the list of functions in :mod:`pynbody.analysis.profile`.

For example, the mass enclosed within a given radius is given by ``mass_enc``:

.. ipython::

    @suppress
    In [4]: plt.clf()

    In [4]: plt.plot(p['rbins'].in_units('kpc'), p['mass_enc'], 'k')

    @savefig profile_encmass.png width=5in
    In [5]: plt.xlabel('$R$ [kpc]'); plt.ylabel(r'$M_{\star}(<R)$')


See the
:class:`~pynbody.analysis.profile.Profile` documentation for a full
list with brief descriptions. You can also check the available
profiles in your session using
:func:`~pynbody.analysis.profile.Profile.derivable_keys`.

.. note::
    You can also define your own profiling functions in your code
    by using the :meth:`Profile.profile_property <pynbody.analysis.profile.Profile.profile_property>`
    decorator; these become available in just the same way as the built-in profiling functions.
    If you wish to do this, the best place to start is by studying the implementation
    of the existing profile properties in the :mod:`~pynbody.analysis.profile` module.

Surface brightnesses
^^^^^^^^^^^^^^^^^^^^

Some of the derivable quantities take parameters. For example, surface brightness
profiles are given by ``sb`` and on consulting the :meth:`docstring <pynbody.analysis.profile.sb>`,
this turns out to take the band as an input. Parameters are passed in to the string using
commas. For example, to get the Johnson U-band surface brightness profile, we ask for ``sb,u``,
or for R-band ``sb,r``:

.. ipython::

    @suppress
    In [4]: plt.clf()

    In [4]: plt.plot(p['rbins'].in_units('kpc'), p['sb,u'], 'b', label="U band");
       ...: plt.plot(p['rbins'].in_units('kpc'), p['sb,r'], 'r', label="R band");

    @savefig profile_mags.png width=5in
    In [5]: plt.xlabel('$R$ [kpc]'); plt.ylabel(r'SB/mag/arcsec$^2$');
       ...: plt.legend()

.. note::
    Surface brightnesses are calculated using SSP tables described further in the
    :mod:`~pynbody.analysis.luminosity` module.


Rotation curves
^^^^^^^^^^^^^^^

Another useful special quantity is the rotation curve, which can be calculated using
the ``v_circ`` key:


.. ipython::

 @suppress
 In [1]: plt.clf()

 In [1]: p_dm = pynbody.analysis.profile.Profile(h[0].dm, min=.05, max=50, type = 'log')

 In [2]: p_gas = pynbody.analysis.profile.Profile(h[0].gas, min=.05, max=50, type = 'log')

 In [3]: p_all = pynbody.analysis.profile.Profile(h[0], min=.05, max=50, type = 'log')

 In [4]: for prof, name in zip([p_all, p_dm, p, p_gas],['total', 'dm', 'stars', 'gas']):
    ...:     plt.plot(prof['rbins'], prof['v_circ'], label=name)

 In [5]: plt.xlabel('$R$ [kpc]');

 In [6]: plt.ylabel('$v_{circ}$ [km/s]');

 @savefig vcirc_profiles.png width=5in
 In [5]: plt.legend()

As the above example makes clear, the circular velocity is estimated from the gravitational force
generated by particles known to the profile object, rather than the entire snapshot.


Calculating Derivatives and Dispersions
---------------------------------------

You can calculate derivatives of profiles automatically. For instance,
you might be interested in d phi / dr if you're looking at a
disk. This is as easy as attaching a ``d_`` to the profile name. For
example:

.. ipython::

   In [6]: p_all = profile.Profile(s, rmin='.01 kpc', rmax='250 kpc')

   In [6]: p_all['pot'][0:10] # returns the potential profile

   In [7]: p_all['d_pot'][0:10] # returns d phi / dr from p["phi"]

Similarly straightforward is the calculation of dispersions and
root-mean-square values. You simply need to attach a ``_disp`` or
``_rms`` as a suffix to the profile name. To get the stellar velocity
dispersion:

.. ipython:: python

    @suppress
    plt.clf()

    plt.plot(p['rbins'].in_units('kpc'), p['vr_disp'].in_units('km s^-1'), 'k')

    @savefig profile_fig2.png width=5in
    plt.xlabel('$R$ [kpc]'); \
    plt.ylabel('$\sigma_{r}$')


In addition to doing this by hand, you can make a
:class:`~pynbody.analysis.profile.QuantileProfile` that can return any
desired quantile range. By default, this is the mean +/- 1-sigma:

.. ipython::

    In [5]: p_quant = profile.QuantileProfile( h[0].s, rmin = '0.1 kpc', rmax = '50 kpc')

    In [6]: plt.clf(); plt.plot(p_quant['rbins'], p_quant['feh'][:,1], 'k')

    In [6]: plt.fill_between(p_quant['rbins'], p_quant['feh'][:,0], p_quant['feh'][:,2], color = 'Grey', alpha=0.5)

    @savefig profile_quant.png width=5in
    In [6]: plt.xlabel('$R$ [kpc]'); plt.ylabel('[Fe/H]')



Vertical Profiles
-----------------

For analyzing disk structure, it is frequently useful to have a
profile in the z-direction. This is done with the
:class:`~pynbody.analysis.profile.VerticalProfile` which behaves in
the same way as the :class:`~pynbody.analysis.profile.Profile`. Unlike
in the basic class, you must specify the radial range and maximum z to
be used:

.. ipython::

   In [5]: p_vert = profile.VerticalProfile( h[0].s, '3 kpc', '5 kpc', '5 kpc')

   In [5]: plt.clf(); plt.plot(p_vert['rbins'].in_units('pc'), p_vert['density'].in_units('Msol pc^-3'),'k')

   @savefig profile_fig5.png width=5in
   In [5]: plt.xlabel('$z$ [pc]'); plt.ylabel(r'$\rho_{\star}$ [M$_{\odot}$ pc$^{-3}$]')




Profiles with arbitrary x-axes
------------------------------

Radial profiles are nice, but sometimes we want a profile using a
different quantity on the x-axis. We might want to know, for example, how the mean
metallicity varies as a function of age in the
stars. :class:`~pynbody.analysis.profile.Profile` by default uses either the 3D or
xy-plane radial distance, depending on
the value of ``ndim``. But we can specify a different function using the
``calc_x`` keyword. Often these are simple so a lambda function can be
used (e.g. if we just want to return an array) or can also be more
complicated functions. For example, to make the profile of stars in
halo 0 according to their age:

.. ipython::

   In [6]: s.s['age'].convert_units('Gyr')

   In [5]: p_age = profile.Profile( h[0].s,
      ...:                          calc_x = lambda x: x.s['age'],
      ...:                          rmax = '10 Gyr' )

   In [6]: plt.clf(); plt.plot(p_age['rbins'], p_age['feh'], 'k', label = 'mean [Fe/H]')

   In [6]: plt.plot(p_age['rbins'], p_age['feh_disp'], 'k--', label = 'dispersion')

   In [6]: plt.xlabel('Age [Gyr]'); plt.ylabel('[Fe/H]')

   @savefig profile_fig4.png width=5in
   In [6]: plt.legend()
