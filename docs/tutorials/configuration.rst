.. configuration tutorial

.. _configuration:

Configuring pynbody
===================

Pynbody uses a flexible configuration system through the standard
Python ``ConfigParser`` class. The
default configuration is stored in ``default_config.ini`` in the
pynbody installation directory.

The options can be overriden by placing a ``.pynbodyrc`` file in your home
directory, or a ``config.ini`` file in the installed package directory, and
specifying different values for configuration parameters to be overriden. These
files do not need to have every option, only the ones you want to change
relative to the ``default_config.ini``.

The configuration files set up basic things like particle family names,
whether to use mutlithreading, default base units etc. The default
configuration file is reproduced verbatim below for reference, allowing
you to see what options can be changed.

Most of the options are
explained in the file itself, and in order to use a different default,
you simply override the option in ``.pynbodyrc``. For example, to reduce
the number of CPU cores that ``pynbody``  uses, the following can be
placed in ``.pynbodyrc``:

::

   [general]

   number_of_threads: 2


For more information on threading, see :ref:`threading`.

Some options can also be changed at runtime. You can check which ones
with

.. ipython::

   In [3]: pynbody.config

For example, to change how many particles are being used to estimate sph
kernel quantities, you can set

.. ipython::

   In [4]: pynbody.config['sph']['smooth-particles'] = 128

Default configuration
---------------------

The default configuration file for pynbody is shown below.

.. literalinclude:: ../../pynbody/default_config.ini
   :language: ini
