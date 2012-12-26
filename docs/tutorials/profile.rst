.. profile tutorial


Profiles in Pynbody
===================


Radial Density Profile 
----------------------
Simple example for calculating a density profile: 

.. plot:: tutorials/example_code/density_profile.py
   :include-source:



The Profile Class 
-----------------

The :func:`~pynbody.analysis.profile.Profile` class is meant to be a
general-purpose class to satisfy all simulation profiling
needs. Profiles are the most elementary ways to begin analyzing a
simulation, so the :func:`~pynbody.analysis.profile.Profile` class is
designed to be an extension of the syntax implemented in the
:func:`~pynbody.snapshot.SimSnap` class and its derivatives.

The constructor only returns a
:func:`~pynbody.analysis.profile.Profile` instance and defines the
bins etc. Importantly, it also stores lists of particle indices
corresponding to each bin, so you can easily identify where the
particles belong.

With the default parameters, the profile is made in the xy-plane. To
make a spherically-symmetric 3D profile, specify ``ndim=3`` when
creating the profile. 

Automatically-generated profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many profiling functions are already implemented -- see the
:func:`~pynbody.analysis.profile.Profile` documentation for a full
list. Additionally, *any* array can be 'profiled'. For example, if
[Fe/H] is a derived field 'feh', then we can plot a metallicity
profile:

>>> plt.plot(ps['rbins'],ps['feh'])

If the array doesn't exist but is deriveable (check with
``s.s.derivable_keys()``), it is automatically calculated.
