.. profile tutorial



Profiles in Pynbody
===================

The Profile lass is meant to be a general-purpose class to satisfy all
simulation profiling needs. Profiles are the most elementary ways to
begin analyzing a simulation, so the Profile class is designed to be
an extension of the syntax implemented in the SimSnap class and its
derivatives.

The constructor only returns a Profile instance and defines the bins
etc. Importantly, it also stores lists of particle indices
corresponding to each bin, so you can easily identify where the
particles belong.

Radial Density Profile 
----------------------
Simple example for calculating a density profile: 

.. plot:: tutorials/example_code/density_profile.py
   :include-source:


.. ipython::
   In [6]: s = pynbody.load('/home/itp/roskar/pynbody_src/nose/testdata/g15784.lr.01024.gz')
   In [7]: h = s.halos()
  
