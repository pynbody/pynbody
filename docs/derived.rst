.. _derived:

Automatically Derived Arrays
============================

`Pynbody` includes a system which automatically derives one array from
others. The idea is that this system

  (1) permits analysis code to assume the existence of arrays
  whether or not they exist in the file;

  (2) allows arrays to be kept up-to-date when they depend on other
  arrays which are changing.


Built-in derived arrays
-----------------------

The quantities listed under :mod:`~pynbody.derived` are calculated for
all simulation types. If you want, for example, to calculate the
specific kinetic energy, you can just access the ``ke`` array and
pynbody will calculate it:

.. ipython:: 
 
   In [3]: import pynbody

   In [4]: s = pynbody.load('testdata/g15784.lr.01024.gz')

   In [5]: s['ke']

Note that you cannot directly alter a derived array:

.. ipython::

   In [5]: s['ke'][0] = 0

But the array updates itself when one of its dependencies changes:

.. ipython::

   In [5]: s['vel']*=2

   In [5]: s['ke']

The recalculation only takes place when required -- in the example
above the ``ke`` array is stored (just like a normal array) until the velocity array is updated,
at which point it is deleted. When you ask for the ``ke`` array again,
it is recalculated. 

This is why you're not allowed to change values in a derived array --
your changes could easily be lost, leading to confusing bugs. However,
if you want to make a derived array back into a normal array you can
do so.

.. ipython::

   In [5]: s['ke'].derived = False

   In [5]: s['ke'][0] = 0

   In [5]: s['ke']


At this point, however, you've taken full responsibility for that
array, so if you update its dependencies the framework *won't* help
you out any longer:

.. ipython::

   In [5]: s['vel']*=2

   In [5]: s['ke']


To get the framework back on your side, you can delete the modified
array:

.. ipython::

   In [5]: del s['ke']

   In [5]: s['ke']


Defining your own deriving functions
------------------------------------

You can easily define your own derived arrays. The easiest way to do
this is using the decorator ``pynbody.derived_array``. An example is
given in the :ref:`data access tutorial <create_arrays>`.

When your function is called, the framework monitors any arrays it
retrieves from the simulation. It automatically marks the accessed
arrays as dependencies for your function. 


Built-in derived arrays for all snapshot classes
------------------------------------------------

.. automodule:: pynbody.derived
   :members: 


tipsy 
-----
   
.. note:: describe how abundances are calculated, give references!

.. autofunction:: pynbody.tipsy.HII
.. autofunction:: pynbody.tipsy.HeIII 
.. autofunction:: pynbody.tipsy.ne
.. autofunction:: pynbody.tipsy.hetot
.. autofunction:: pynbody.tipsy.hydrogen
.. autofunction:: pynbody.tipsy.feh
.. autofunction:: pynbody.tipsy.oxh
.. autofunction:: pynbody.tipsy.ofe
.. autofunction:: pynbody.tipsy.mgfe
.. autofunction:: pynbody.tipsy.nefe
.. autofunction:: pynbody.tipsy.sife
.. autofunction:: pynbody.tipsy.c_s
.. autofunction:: pynbody.tipsy.c_s_turb
.. autofunction:: pynbody.tipsy.mjeans
.. autofunction:: pynbody.tipsy.mjeans_turb
.. autofunction:: pynbody.tipsy.ljeans
.. autofunction:: pynbody.tipsy.ljeans_turb


gadget
------

No special derived quantities at the moment.


Ramses
------

.. autofunction:: pynbody.ramses.mass

