.. Last checked by AP: 18 Mar 2024

.. pynbody tutorials main index

.. _tutorials:

Pynbody Tutorials
=================

Here you will find tutorials that illustrate the use of pynbody. The
:ref:`walkthroughs` demonstrate pynbody's most important functionality. Then there
are :ref:`deeper-dives` with some more detail of how pynbody's core
library works. Finally, some :ref:`advanced_topics` are covered.

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
   Images <images>
   Profiles <profile>

.. _deeper-dives:

Deeper dives
------------

.. toctree::
   :maxdepth: 1

   Reading snapshots <data_access>
   Sub-views & filters <filters>
   Halos & groups <halos>
   Halo mass function <hmf>
   Linking snapshots <bridge>


.. _advanced_topics:

Advanced topics
---------------

.. toctree::
   :maxdepth: 1

   Derived quantities <derived>
   Configuration <configuration>
   Parallelism <parallelism>
   Performance <performance>
   Changes in version 2 <changes_v2.rst>
