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

.. _obtaining_testdata:

Obtaining test data
-------------------

Many of the tutorials below use the same dataset that pynbody is tested on. While they are easily adapted for your
own data, you can also download and unpack the test data to replicate the examples exactly. To do this, you can either
manually download the tarballs from `zenodo <https://zenodo.org/doi/10.5281/zenodo.12552027>`_ and unpack them
to a directory of your choice, or automatically download using pynbody's built-in testdata downloader:

.. code:: python

    import pynbody.test_utils
    pynbody.test_utils.precache_test_data()

Note that pynbody automatically creates a folder called ``testdata`` in this case, and the tutorials assume
that you have put the data in this folder.

.. _walkthroughs:

Walkthroughs
------------

.. toctree::
   :maxdepth: 1

   Quick-start <quickstart>
   Reading snapshots <data_access>
   Sub-views & filters <filters>
   Halos & groups <halos>
   Linking snapshots <bridge>

.. _cookbook:

Cookbook/Recipes
----------------

.. toctree::
   :maxdepth: 1

   Profiles <profile>
   Images <images>
   Halo mass function <hmf>


.. _advanced_topics:

Advanced topics
---------------

.. toctree::
   :maxdepth: 1

   Derived quantities <derived>
   Configuration <configuration>
   Parallelism <parallelism>
   Performance <performance>
