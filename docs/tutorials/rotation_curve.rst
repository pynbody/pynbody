.. rotation curve



Plotting Rotation Cuves
=======================

Rotation curves are calculated and plotted using the :func:`~pynbody.analysis.profile.Profile`
object. For a tutorial on profiles see :doc:`profile`.

.. plot:: tutorials/example_code/rotcurve.py
   :include-source:


Speeding up the Gravity Calculation in Pynbody
----------------------------------------------

The rotation curve is calculated by calculating the forces in a
plane. The force calculation is a direct :math:`N^2` calculation, so
it takes a while and it is therefore done in ``C``. It is even faster if
you use the parallel Open-MP version. This was installed automatically
if pynbody detected an Open-MP C compiler during setup. To see whether
this happened or not, you can ask how many cores your machine has:

.. ipython::

 In [4]: import pynbody

 In [4]: pynbody.openmp.get_cpus()

If this returns 1 but you have more than one core, pynbody has not
been installed with OpenMP support (check your :ref:`compiler
<pynbody-installation-must-haves>`, especially on Mac OS - then `raise
an issue <https://github.com/pynbody/pynbody/issues>`_ if you can't
fix the problem) . If it returns the number of cores on your
machine, you're in luck.

Assuming OpenMP support is enabled, the actual number of cores used by
pynbody is determined by the configuration option
``number_of_threads``, which is the number of CPUs detected on your
machine by default. If you want to reduce this (e.g. you are running on
a login node or have multiple analyses going on in parallel), you can
specify the number of cores explicitly:

.. ipython::

 In [5]: pynbody.config['number_of_threads'] = 2

Now all gravity calculations will use the parallel gravity calculation
on only 2 cpus. Note that the number of threads you specify in this option
will also be the default for other routines, such as
:func:`pynbody.plot.sph.image`. See :ref:`threads`.

If you want to make your configuration changes permanent, create a
file called ``.pynbodyrc`` in your home directory with the lines

::

   [general]
   number_of_threads: 2


See  :ref:`configuration` for more information.
