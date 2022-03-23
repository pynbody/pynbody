.. pynbody documentation master file, created by
   sphinx-quickstart on Mon Oct  3 11:57:24 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.


Pynbody Documentation
===============================

Welcome to the documentation for `pynbody
<http://pynbody.github.io>`_ -- an analysis package for
astrophysical N-body and Smooth Particle Hydrodynamics
simulations, supporting Python 3.5+. (Old versions are
available, prior to 1.0, that support Python 2.5, 2.6 and 2.7).

We recommend you get started by reading about
:ref:`pynbody-installation` and trying the :ref:`tutorials`. We are
happy to provide further assistance via our
`user group email list
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_.

Where next?
-----------

Consult the :doc:`installation` documentation for instructions on how
to get going. Then you might like to download some `test data
<https://github.com/pynbody/pynbody/releases>`_ and try out the
:ref:`first steps tutorial <snapshot_manipulation>` which gets straight
to some of pynbody's analysis features. Or, if you prefer to learn
a little more of how your data is organized, we also provide a :ref:`data
access walkthrough <data-access>`.

Our full documentation is organized into three sections:

.. toctree::
   :maxdepth: 1

   Installation <installation>
   Tutorials & walkthroughs <tutorials/tutorials>
   Reference <reference/index>


All of the information in the reference guide is also available
through the interactive python help system. In ipython or Jupyter, this is as
easy as putting a ``?`` at the end of a command:

.. ipython::

   In [1]: import pynbody

   In [2]: pynbody.load?



.. _getting-help:

Seeking Further Assistance
---------------------------


If the tutorials and reference documentation don't answer your question,
any problem might be described in :doc:`pitfalls`.

If you still find yourself stuck, don't hesitate to post a message to the
`users group
<https://groups.google.com/forum/?fromgroups#!forum/pynbody-users>`_.
If you have a Google account you can join the groups
easily, but if you don't have one please click on the ``About`` button
of the group you are interested in and contact the owner.

We have found that most development discussion takes place within our
`github issue tracker <https://github.com/pynbody/pynbody/issues>`_ -- if you
encounter a problem, feel free to create an issue there.
We greatly value feedback from users, especially when things are not
working correctly because this is the best way for us to correct
bugs.  This includes any problems you encounter with documentation.

Please note that we adhere to a  `community code of conduct
<https://github.com/pynbody/pynbody/blob/master/CODE_OF_CONDUCT.md>`_,
which you should read and understand before posting to the users list or in a github issue.

If you use the code regularly for your projects, please consider contributing
your code back using a `pull request
<https://help.github.com/articles/using-pull-requests>`_.

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
