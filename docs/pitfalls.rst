.. pitfalls Common Pitfalls


Common Pitfalls
===============

Units - Tipsy param file
^^^^^^^^^^^^^^^^^^^^^^^^
One of the great things about pynbody is that it takes care of units, but
if it can't figure out what units to use, it gets grumpy and gives long errors.

`Always make sure you have a param file in the directory where you are 
analyzing a tipsy file, and make sure that it defines dKpcUnit and dMsolUnit.`

If you are analyzing a DM only simulation, it's probably just easier to play
along and assign units even though they weren't needed for the simulation.

Image width
^^^^^^^^^^^
The `width` keyword for the image function doesn't take units, so you 
need to assign units before you use it.  The easiest way to do this is with

::

 In [1]: s.physical_units()

if your simulation is called ``s``.  If you don't do this, python will likely
try to make an image with no data and tell you about it in a long error message.

