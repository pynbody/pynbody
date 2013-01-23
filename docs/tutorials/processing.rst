.. profile tutorial


Processing Snapshots with Pynbody
=================================

How I use pynbody
-----------------
One typical use for scripting software like python is to automate common
tasks.  I find it extremely useful to run a standard series of analysis routines
to produce small png files on the cluster where I'm running simulations.
That way, I can quickly download the 1 MB of images rather than the 
~1 GB snapshot files to find out how my simulation is progressing.

I also create a `.data` file using the python pickle module.  This includes
fundamental properties of the snapshot like the mass of stars, hot gas,
cold gas, radial profiles, a rotation curve.  That way, when you want to 
make future plots comparing snapshots from different times or different
parameter choices, you only have to use pickle to quickly open the 50 kB
data file instead of opening the 1 GB snapshot file.


doall.py
--------
There is an example script in pynbody/examples called doall.py that does 
a lot of generic analysis of a cosmological galaxy simulation.  To go through
it piece by piece, let's look at the three main sections, the part that 
stores data, the part that makes some standard plots and the part the part
that makes the images: 


Pickling Data
^^^^^^^^^^^^^

.. literalinclude:: example_code/do_data.py

Standard plots
^^^^^^^^^^^^^^

.. literalinclude:: example_code/do_plots.py


Standard Images
^^^^^^^^^^^^^^^

.. literalinclude:: example_code/do_plots.py


