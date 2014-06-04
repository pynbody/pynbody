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

The :func:`~pynbody.analysis.profile.Profile` class is meant to be a
general-purpose class to satisfy (almost) all simulation profiling
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
list. Additionally, *any* existing array can be 'profiled'. For
example, if [Fe/H] is a derived field 'feh', then plotting a
metallicity profile is as simple as: 

>>> plt.plot(p['rbins'],p['feh'])

If the array doesn't exist but is deriveable (check with
``s.derivable_keys()``), it is automatically calculated.

You can define your own profiling functions in your code by using the
``Profile.profile_property`` decorator::

   @pynbody.analysis.profile.Profile.profile_property
    def random(self):
        """
        Generate a random profile
        """
	import numpy as np
        return np.random.rand(self.nbins)


Calculating Derivatives and Dispersions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can calculate derivatives of profiles automatically. For instance,
you might be interested in d phi / dr if you're looking at a
disk. This is as easy as attaching a ``d_`` to the profile name. For
example:

>>> p = pynbody.analysis.profile.Profile(s)
>>> p['phi'] # returns the potential profile
>>> p['d_phi'] # returns d phi / dr from p["phi"]

Similarly straightforward is the calculation of dispersions and
root-mean-square values. You simply need to attach a ``_disp`` or
``_rms`` as a suffix to the profile name:

>>> p['vr_disp']
>>> p['z_rms']


