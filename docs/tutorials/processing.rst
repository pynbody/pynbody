.. processing tutorial


Batch-processing Snapshots with Pynbody
=======================================

Using pynbody as a batch system
-------------------------------

One typical use for scripting software like python is to automate common
tasks.  You can use `pynbody` to run a standard series of analysis
routines -- for instance to produce small png files on the cluster where you're running simulations.
That way, you can just download 1 MB of images rather than the 
~1 GB snapshot files to find out how a simulation is progressing.

Going a step further, you might also create a `.data` file using the
python pickle module, giving fundamental properties of the
snapshot like the mass of stars, hot gas, cold gas, radial profiles, a
rotation curve.  That way, when you want to make future plots
comparing snapshots from different times or different parameter
choices, you only have to use pickle to quickly open the 50 kB data
file instead of opening the 1 GB snapshot file.


doall.py
--------
There is an example script in pynbody/examples called doall.py that does 
a lot of generic analysis of a cosmological galaxy simulation.  To go through
it piece by piece, let's look at the four main sections: the :ref:`dopreamble`
the part that 
:ref:`storesdata`, the part that :ref:`standardplots` and the part
that :ref:`images`: 

.. _dopreamble:

Preamble
^^^^^^^^

.. literalinclude:: example_code/do_preamble.py

Most of the lines in the preamble are commented as to their purpose.  Overall,

#. The appropriate modules are loaded with the import statements
#. The simulation is loaded based on a command line argument
#. If a photogenic file (gasoline thing) is given, the halo that includes those particle ids is loaded.
#. The halo is rotated so the disk is face-on

.. _storesdata:

pickle data
^^^^^^^^^^^

.. literalinclude:: example_code/do_data.py

#. Some special pieces of data about the halo are calculated for storage.
#. The data is saved using the python pickle module (pickle.dump) 

.. note:: The kinematic decomposition section is commented out because it is 
 a little slow.

* The pickled data can be used to make comparisons as in the following
  code

.. literalinclude:: example_code/rcs.py

.. _standardplots:

make standard plots
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: example_code/do_plots.py

* Inside of a ``try: except:`` statement, a series of standard plots are created.  Because of the ``try: except: pass``, if any of them fails, the script will continue.
* For each of these procedures, the filename argument lets you save the plot as the given filename.
* In some of the plots, fixed axis ranges are given so that quick movies can be made by putting all the image frames together.

.. _images:

make standard images
^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: example_code/do_images.py

* The images are similar to the plots in that they get saved to the filename argument.
* One difference is that the images require a width to say how much of the simulation they should include.
* vmin and vmax represent the dynamic range of the image.
* A couple of auxillary quantities are calculated like the enrichment of a certain ion species.

