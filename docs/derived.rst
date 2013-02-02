


Automatically Derived Quantities
================================


These are quantities that are calculated and lazy-loaded automatically
by pynbody. The quantities listed under
:mod:`~pynbody.derived` are calculated for all simulation
types. If you want, for example, to calculate the specific kinetic
energy, you can just access the ``ke`` array and pynbody will calculate it: 

.. ipython:: 
 
   In [3]: import pynbody

   In [4]: s = pynbody.load('testdata/g15784.lr.01024.gz')

   In [5]: s['ke']


.. automodule:: pynbody.derived
   :members: 

tipsy 
-----
   
.. note:: describe how abundances are calculated, give references!


.. autofunction:: pynbody.tipsy.HII
.. autofunction:: pynbody.tipsy.HeIII 
.. autofunction:: pynbody.tipsy.ne
.. autofunction:: pynbody.tipsy.mu
.. autofunction:: pynbody.tipsy.u
.. autofunction:: pynbody.tipsy.p
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

