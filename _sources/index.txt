.. pynbody documentation master file, created by
   sphinx-quickstart on Mon Oct  3 11:57:24 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Pynbody Documentation
===============================

Welcome to the documentation for `pynbody
<http://pynbody.github.io>`_ -- an analysis package for
astrophysical N-body and Smooth Particle Hydrodynamics
simulations. We recommend you get started by reading about
:ref:`pynbody-installation` and trying the :ref:`tutorials`. We are
happy to provide further assistance via our
`user group email list
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_. 


Installation and first steps
----------------------------

Consult the :doc:`installation` documentation for instructions on how
to get going. Then you might like to download some `test data
<https://code.google.com/p/pynbody/downloads/list>`_ and try out the
:ref:`first steps tutorial <snapshot_manipulation>` or a :ref:`lower-level
data access walkthrough <data-access>`.


.. _getting-help: 

Getting Help
------------

Tutorials, Reference Documentation and Online Help
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get started with `Pynbody`, see the tutorials and reference guide
below. All of the information in the reference guide is also available
through the interactive python help system. In ipython, this is as
easy as putting a `?` at the end of a command:

.. ipython::

   In [1]: import pynbody

   In [2]: pynbody.load? 


Common Pitfalls
^^^^^^^^^^^^^^^

We are compiling a list of common problems users might experience. If
you just installed pynbody and upon trying some fancy analysis it
greeted you with a complicated error message, the fix might be
described in :doc:`pitfalls`.


Pynbody Users Group
^^^^^^^^^^^^^^^^^^^

If you find yourself stuck, don't hesitate to post a message to the
`users group
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_. If
you'd like to take part in the development of pynbody, joining the
`developer group
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-dev>`_ is
a good idea. If you have a Google account you can join the groups
easily, but if you don't have one please click on the `About` button
of the group you are interested in and contact the owner.

Reporting Issues
^^^^^^^^^^^^^^^^

We really value feedback from users, especially when things are not
working correctly because this is the best way for us to correct
bugs. Please let us know if you find strange behavior or if you have
ideas for improvements. The best way to do this is via the `users
group
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_
or the `issues page <https://github.com/pynbody/pynbody/issues>`_ on
the `Pynbody Github site
<https://github.com/pynbody/pynbody/>`_. The documentation in particular
is in a state of flux so if you find an error in any of the tutorials
let us know asap. If you use the code regularly for your projects,
consider becoming a contributor and join the `developer group
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-dev>`_


.. _acknowledging-pynbody:

Acknowledging Pynbody in Scientific Publications
------------------------------------------------

Pynbody development is an open-source, community effort. The only way
to make it as robust as possible is to have a wide user-base and this
is only possible by spreading the word. We ask that if you use pynbody
in preparing a scientific publication, you cite it via its
`Astrophysics Source Code Library <http://ascl.net/1305.002>`_ entry
using the following BibTex::

   @misc{pynbody,
     author = {{Pontzen}, A. and {Ro{\v s}kar}, R. and {Stinson}, G.~S. and {Woods},
        R. and {Reed}, D.~M. and {Coles}, J. and {Quinn}, T.~R.},
     title = "{pynbody: Astrophysics Simulation Analysis for Python}",
     note = {Astrophysics Source Code Library, ascl:1305.002},
     year = 2013
   }

You may also mention it either as a footnote in the text or in the
acknowledgments section with a statement like:

**We made use of pynbody (https://github.com/pynbody/pynbody) in our analysis
for this paper.**


Tutorials
---------

The tutorials are not a complete guide to Pynbody, but they will help
new users get started, as well as provide some more in-depth
information for the seasoned users.

.. toctree::
   :maxdepth: 2

   Pynbody tutorials <tutorials/tutorials>   


Reference
-------------

.. toctree::
   :maxdepth: 3
   
   Simulation loaders <loaders>
   Essential generic modules <essentials>   
   Convenience modules <convenience>
   Derived quantities <derived>
   Analysis modules <analysis>
   Plotting modules <plot>


