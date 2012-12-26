.. profile tutorial



Profiles in Pynbody
===================

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

Radial Density Profile 
----------------------
Simple example for calculating a density profile: 

.. plot:: tutorials/example_code/density_profile.py
   :include-source:


  
