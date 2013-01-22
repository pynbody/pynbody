.. pynbody documentation master file, created by
   sphinx-quickstart on Mon Oct  3 11:57:24 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Pynbody Documentation
===============================

Welcome to the documentation for `pynbody
<http://code.google.com/p/pynbody/>`_ -- an analysis package for
astrophysical N-body and Smooth Particle Hydrodynamics
simulations. We recommend you get started by reading about
:ref:`pynbody-installation` and trying the :ref:`tutorials`. We are
happy to provide further assistance via our
`user group email list
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_. 


Installation
------------

Consult the :doc:`installation` documentation for instructions on how
to get going.


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
bugs. This is a community effort so please let us know if you find
strange behavior or if you have ideas for improvements. The best way
to do this is via the `Issues page
<http://code.google.com/p/pynbody/issues/list>`_ on the `Pynbody
Google Code site <http://code.google.com/p/pynbody/>`_. If you use the
code regularly for your projects, consider becoming a contributor!


.. _acknowledging-pynbody:

Acknowledging Pynbody in Scientific Publications
------------------------------------------------

Pynbody development is an open-source, community effort. The only way
to make it as robust as possible is to have a wide user-base and this
is only possible by spreading the word. We currently do not have a
paper that you could cite, but we ask that if you use pynbody in
preparing a scientific publication, you mention it either as a
footnote in the text or in the ackowledgments section. Thank you.


Tutorials
------------

.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/tutorials>   


Reference
-------------

.. toctree::
   :maxdepth: 3

   Simulation loaders <loaders>
   Essential generic modules <essentials>   
   Convenience modules <convenience>
   Analysis modules <analysis>
   Plotting modules <plot>


