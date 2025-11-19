.. _hmf_tutorial:


Halo Mass function
===================

This recipe makes use of the module :mod:`~pynbody.analysis.halo` to generate the halo mass function of a given snapshot
and compare it to a theoretical model.

.. note::

 The halo mass function code in pynbody was implemented in 2012 when there
 were no other python libraries that could calculate theoretical HMFs.
 Since then, cosmology-focussed libraries such as `hmf <https://hmf.readthedocs.io/en/latest/index.html>`_,
 `Colossus <https://bdiemer.bitbucket.io/colossus/index.html>`_
 and `CCL <https://ccl.readthedocs.io/en/latest/>`_ have been developed.
 For precision cosmology applications, we recommend using these libraries.
 The functionality here is retained for quick cross-checks of simulations.


Calculating a theoretical halo mass function
--------------------------------------------

We will start by loading a snapshot data.


.. ipython::

  In [1]: import pynbody
     ...: import matplotlib.pyplot as p
     ...: s = pynbody.load('testdata/tutorial_gadget/snapshot_020')
     ...: s.physical_units()

To define the expected halo mass function, we need to make sure that the cosmology is well set. Some cosmological
parameters are carried by the snapshot itself. However, others like ``sigma_8`` are usually not, so you might want to
specify it to the value used to create the simulation. In this particular file, we can check the
density and Hubble parameters then set ``sigma8``.

.. ipython::

  In [1]: s.properties['omegaM0'], s.properties['omegaL0'], s.properties['h']

  In [2]: s.properties['sigma8'] = 0.8288


You can generate an HMF using the :func:`~pynbody.analysis.hmf.halo_mass_function` function as follows.

.. ipython::

  In [3]: m, sig, dn_dlogm = pynbody.analysis.hmf.halo_mass_function(s, log_M_min=10, log_M_max=15, delta_log_M=0.1,
     ...:                                                            kern="REEDU")

By specifying ``kern="REEDU"``, we are asking for a Reed (2007) mass function. Other options are
described in the documentation of :func:`~pynbody.analysis.hmf.halo_mass_function`.

Let's inspect the output:

.. ipython:: python

  p.plot(m, dn_dlogm)
  p.loglog()
  p.xlabel(f"$M / {m.units.latex()}$")
  @savefig theory_hmf.png width=6in
  p.ylabel(f"$dn / d\\log M / {dn_dlogm.units.latex()}$")



Obtaining the binned halo mass function from the simulation
-----------------------------------------------------------

We now generate the HMF from binned halo counts in a simulation volume.

This method calculates all halo masses contained in the snapshot
and bin them in a given mass range. The number count is then normalised by the simulation volume to obtain
the snapshot HMF. We can get a sense of error bars from the number count in each bin assuming
a Poissonian distribution:

.. ipython::

  In [2]: bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(s,
     ...:                        log_M_min=10, log_M_max=15, delta_log_M=0.1, )


We are now ready to compare the two results on a plot:

.. ipython::

  In [2]: plt.errorbar(bin_center, bin_counts, yerr=err, fmt='o',
     ...:              capthick=2, elinewidth=2, color='darkgoldenrod')


  @savefig hmf_comparison.png width=6in
  In [2]: plt.xlim(1e10, 1e15)

The agreement is pretty good. Note that in generating the empirical halo mass function above,
Pynbody has summed the mass of particles in each halo to get the halo mass. This may not
be what you want, especially e.g. if you want to compare with virial masses rather than
bound masses. Furthermore, summing over particles for each halo can be slow for large simulations.
For all these reasons, if the halo finder provides pre-calculated masses you can use those
instead by passing them to the ``mass_property`` argument of
:func:`~pynbody.analysis.hmf.simulation_halo_mass_function`.
First, check the available properties for your halo catalogue:

.. ipython::

    In [2]: s.halos()[0].properties.keys()


Here we can see SubFind calculated various mass definitions like ``mmean_200``,
``mcrit_200`` etc. The particular properties available will depend on your halo finder.
Let's use ``mmean_200`` as another comparison with the HMF:

.. ipython::

    In [2]: bin_center, bin_counts, err = pynbody.analysis.hmf.simulation_halo_mass_function(s,
       ...:                        log_M_min=10, log_M_max=15, delta_log_M=0.1,
       ...:                        mass_property='mmean_200')

    In [2]: plt.errorbar(bin_center, bin_counts, yerr=err, fmt='o',
       ...:              capthick=2, elinewidth=2, color='k', alpha=0.5)

    @savefig hmf_comparison_finder_mass.png width=6in
    In [2]: plt.xlim(1e10, 1e15)

You can see that this agrees well. The slight change is expected because of the change in
halo mass definition from a FoF mass to a spherical overdensity mass. The disagreement
at low masses is due to the finite resolution of the simulation.
