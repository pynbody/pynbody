.. picture tutorial

Pictures in Pynbody
===================

Density Slice
-------------
The essential kind of image -- a density slice:

.. plot:: tutorials/example_code/density_slice.py
   :include-source:

Integrated Density
------------------
Line-of-sight averaged density map:

.. plot:: tutorials/example_code/density_integrated.py
   :include-source:

Temperature Slice
-----------------
Simple example for displaying a slice of some other quantity (Temperature 
in this case)

.. plot:: tutorials/example_code/temperature_slice.py
   :include-source:

Velocity Vectors
----------------
It is also straightforward to obtain an image with velocity 
vectors or flow lines overlaid:

.. plot:: tutorials/example_code/velocity_vectors.py
   :include-source:

Multi-band Images of Stars
--------------------------

You can create visualizations of the stellar distribution using
synthetic colors in a variety of bands:

.. plot:: tutorials/example_code/star_render.py
   :include-source:


Creating images using :func:`~pynbody.plot.sph.image`
-----------------------------------------------------

The :func:`~pynbody.plot.sph.image` function is a general purpose function 
for creating an x-y map of a value from your :func:`~pynbody.snapshot.SimSnap` 
object. Under the hood, the function calls one two SPH kernels (written in c) 
to calculate the intensity values of whatever value you're plotting - a 2d 
kernel for integrated maps, and a 3d kernel for slices. Both kernels require 
the use of a kd-tree to perform an SPH smooth, so you will notice that the 
first time :func:`~pynbody.plot.sph.image` is called, it creates a kd-tree. 
Subsequent calls on the same data set should use the already created tree, 
and thus should be faster.

:func:`~pynbody.plot.sph.image` returns an x,y array representing pixel 
intensity. The function also displays the image with automatically created
axes and a colorbar. However, one can use the x-y array and `plt.imshow() <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow>`_ 
(how do I link to matplotlib functions?) to create your own plot.


Common issues
-------------

:func:`~pynbody.plot.sph.image` is prone to a number of common errors 
when being used by new users. Probably the most common is

::

   ValueError: zero-size array to minimum.reduce without identity

This can come about in a number of circumstances, but essentially it 
means that there were not enough particles in the region that was being 
plotted. It could be due to no/bad centering, passing in a very small/empty 
:func:`~pynbody.snapshot.SubSnap` object, or bad units (units being an issue should 
no longer be an issue. In older versions of pynbody, the width parameter 
assumed kpc, so if the simulation distances were in e.g. "au", this could 
cause a problem).

Another common error is the following:

:: 

   TypeError: 'int' object does not support item assignment

which occurs when the returned image from the kernel is a singular value 
rather than an array. In this case, the issue was because the kernel did 
not complete because of attempting to plot a value for the whole 
:func:`~pynbody.snapshot.Snapshot` object rather than a specific family (such 
as gas). In this case, the "smooth" array needed to be deleted before another 
image could be produced because SPH needed to resmooth with the new dark and 
star particles.
