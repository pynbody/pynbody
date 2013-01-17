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
it takes a while and it is therefore done in `C`. It is even faster if
you use the parallel Open-MP version. This requires `Cython` and a
multi-core machine (which includes all modern computers...). 

Check if you have `Cython` installed: 

.. ipython::

 In [1]: import cython
   
If this doesn't raise an error you're good to go! If this fails,
you'll need to `install Cython <http://cython.org/>`_ first.

Once you have `Cython`, the parallel version should be compiled and
installed automatically. Now, to use the parallel version in the
gravity calculation for the rotation curve, you need to make sure that
your pynbody options are set correctly. You can check the current
configuration like this:

.. ipython:: 

 In [2]: import pynbody

 In [3]: pynbody.config


Here we see two options: ``gravity_calculation_mode = 'direct'`` and
``number_of_threads = 4``. The second option tells pynbody to use 4
threads, but the first instructs it to use the serial version of the
gravity code. We can fix this, and if your machine has more cores you can tell it to use them: 

.. ipython:: 

 In [4]: pynbody.config['gravity_calculation_mode'] = 'direct_omp'

 In [5]: pynbody.config['number_of_threads'] = 30

Now all gravity calculations will use the parallel gravity calculation
on 30 cpus. Note that the number of threads you specify in this option
will also be the default for other routines, such as
:func:`pynbody.plot.sph.image`.

If you want to make your configuration changes permanent, create a file called ``.pynbodyrc`` in your home directory with the lines

:: 

   gravity_calculation_mode: direct_omp
   number_of_threads: 30


See the file ``default_config.ini`` in the pynbody source distribution
for other options.
