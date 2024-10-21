.. pynbody documentation master file, created by
   sphinx-quickstart on Mon Oct  3 11:57:24 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

.. warning::

 You are looking at the documentation for pynbody v2, which is currently in beta.
 To install pynbody v2 beta, use ``pip install --pre pynbody``.  Documentation for
 v1 remains available at https://pynbody.github.io/pynbody/ .

Welcome to the documentation for `pynbody
<http://pynbody.github.io>`_ -- an analysis package for
astrophysical N-body and Smooth Particle Hydrodynamics
simulations, supporting Python 3 with minor version support
adhering roughly to the `SPEC0 <https://scientific-python.org/specs/spec-0000/>`_ policy.

Installation should be as simple as

.. code-block:: bash

   pip install pynbody

but if you run into trouble, try the :doc:`installation` guide.

Once installed, we recommend you get started by trying the :ref:`tutorials`.

.. _getting-help:

Support
-------

Pynbody is a complex project maintained by a small team of scientists, and our dayjobs is doing science!
We do our best to provide support, and we greatly appreciate feedback and bug reports. If you encounter any problems,
please consider `opening an issue on github <https://github.com/pynbody/pynbody/issues/new/choose>`_ or posting to our
`email list <https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_. To help us help you, when asking for
assistance please provide a simple, minimal python script that we can use to reproduce a bug, inaccuracy or situation.
Preferably reproduce the bug using pynbody's own test data (as used by the tutorials,
and downloadable from `zenodo <https://zenodo.org/doi/10.5281/zenodo.12552027>`_).
If this impossible, provide us a pointer to another file that we can use to reproduce the problem.

Please note that we adhere to a  `community code of conduct
<https://github.com/pynbody/pynbody/blob/master/CODE_OF_CONDUCT.md>`_,
which you should read and understand before posting to the users list or in a github issue.

Open science relies on good will and reciprocity. Please strongly consider contributing any enhancements or fixes of
your own back to the project using a `pull request
<https://help.github.com/articles/using-pull-requests>`_.



.. _acknowledging-pynbody:

Acknowledging Pynbody in Scientific Publications
------------------------------------------------

Pynbody development is an open-source, community effort.We ask that if you use pynbody
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



Where next?
-----------

Consult the :doc:`installation` documentation for instructions on how
to get going. Then you might like to download some `test data
<https://github.com/pynbody/pynbody/releases>`_ and try out the
:ref:`quick-start tutorial <quickstart>` which gets straight
to some of pynbody's analysis features. Or, if you prefer to learn
a little more of how your data is organized, we also provide a :ref:`data
access walkthrough <data-access>`.

Our full documentation is organized into three sections:

.. toctree::
   :maxdepth: 3

   Tutorials & walkthroughs <tutorials/tutorials>
   Reference <reference/index>
   Installation <installation>
