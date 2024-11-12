.. _changes::

Changes in version 2
====================


Version 2 of *pynbody* is the first ever release to break full backwards compatibility. Most code for
pynbody 1.6 will also work with pynbody 2.0, but there are some changes that may require you to
modify code. This document summarises the most important changes.

Rationale
---------

The changes in pynbody 2.0 are designed to make the codebase more maintainable. The essential principle
was to remove rarely-used code that was adding complication -- to do less, but do it better.
Breaking changes are never pleasant, but in its 15-year history, pynbody has never had a major version
increment like this. We hope that the changes will make the codebase more sustainable in the long term.

Key changes
-----------

- The documentation, including tutorials, has been completely overhauled and updated.
- The :mod:`~pynbody.halo` subpackage has been completely re-implemented, for greater consistency across
  formats. For many simple uses, old code will continue to work. However, please read the change notice
  in :mod:`~pynbody.halo` for important changes. This especially affects certain AHF catalogues which used
  to be given an inconsistent halo numbering scheme. Backwards compatibility can be achieved there by
  passing the correct flag. There is also now a format-independent way to describe parent/child relationships
  in catalogs, which is described in the :ref:`halo_tutorial` tutorial.
- The :mod:`~pynbody.plot` subpackage has been streamlined, with a number of trivial routines removed, and
  options to save files etc removed. These are much better handled by matplotlib directly.
  See the :mod:`~pynbody.plot` documentation for details.
- The :mod:`~pynbody.array` subpackage now publishes a number of functions that were previously hidden,
  specifically enabling the use of shared memory in complex analysis pipelines across multiple processors. For
  more information see :ref:`using_shared_arrays`.
- INCOMPLETE - MORE TO COME HERE
