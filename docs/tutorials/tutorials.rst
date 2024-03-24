.. Last checked by AP: 18 Mar 2024

.. pynbody tutorials main index

.. _tutorials:

Pynbody Tutorials
=================

Here you will find tutorials that illustrate the use of pynbody. The
:ref:`walkthroughs` demonstrate pynbody functionality through an
interactive session in the ipython shell that you can follow along
using either one of the outputs from the bundled test-data or one of
your own simulations. To complement the walkthroughs, we also provide
a quick-start `Jupyter notebook
<https://github.com/pynbody/pynbody/blob/master/examples/pynbody_demo.ipynb>`_. The
:ref:`cookbook` tutorials are more goal-oriented: they provide a
script that can be used with only minor modifications to immediately
produce a result (i.e. make an image). They also include, however, a
somewhat more involved discussion of more advanced options and common
pitfalls. Finally, the :ref:`advanced_topics` section is meant to
provide a more in-depth look at the inner-workings of the code.

.. _walkthroughs:

Walkthroughs
------------

.. toctree::
   :maxdepth: 1

   Quick-start <snapshot_manipulation>
   Reading data <data_access>
   Filtering data <filters>
   Linking snapshots <bridge>

.. _cookbook:

Cookbook/Recipes
----------------

.. toctree::
   :maxdepth: 1

   Profiles <profile>
   Rotation curves <rotation_curve>
   Images <pictures>
   Halos <halos>
   Halo mass function <hmf>


.. _advanced_topics:

Advanced topics
---------------

These tutorials are likely to be of interest only in special cases.

.. toctree::
   :maxdepth: 1

   Performance <performance>
   Configuration <configuration>
   Threading/multiprocessing <threads>
