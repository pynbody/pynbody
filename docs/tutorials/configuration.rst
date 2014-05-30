.. configuration tutorial

.. _configuration:

Configuring pynbody
===================

Pynbody uses a flexible configuration system through the standard
Python `ConfigParser
<https://docs.python.org/2/library/configparser.html>`_ class. The
default configuration is stored in ``default_config.ini`` in the
pynbody installation directory. The options can be overriden by
placing a ``.pynbodyrc`` file in the home directory and specifying
different values for configuration parameters. The configuration file
sets up basic things like particle family names, whether to use
mutlithreading, default base units etc.

To find ``default_config.ini``, you can type

.. ipython:: 

   In [1]: import pynbody

   In [2]: pynbody.__path__

to find the `pynbody` installation directory. The
``default_config.ini`` can be found there. Most of the options are
explained in the file itself, and in order to use a different default,
you simply override the option in ``.pynbodyrc``. For example, if I
want to change how many threads `pynbody` should use, I would put in
``.pynbodyrc``:

::

   [general]

   number_of_threads: 10



Some options can also be changed at runtime. You can check which ones
with

.. ipython::

   In [3]: pynbody.config

If you wanted to, for example, change how many particles are being
used to estimate sph kernel quantities, you could do

.. ipython::
   
   In [4]: pynbody.config['sph']['smooth-particles'] = 64

   
