"""
pynbody
=======

A light-weight, portable, format-transparent analysis framework
for N-body and SPH astrophysical simulations.

Getting help
------------

Documentation for `pynbody` is maintained in two forms.

 1. Tutorial-style documentation, which can be
    accessed :ref:`here <tutorials>`. If you are viewing this help from
    within python, please visit http://www.itp.uzh.ch/~roskar/pynbody/docs/.
 2. Reference documentation within the code, which can be accessed
    using the standard python help() function, the ipython ? operator,
    or online.

There is also a `user group <https://groups.google.com/forum/?fromgroups=#!forum/pynbody-users>`_
where the developers are happy to help with any problems you encounter.



What's available
----------------

Pynbody handles Gadget, Gadget-HDF, Tipsy, Nchilada and Ramses files. To load
any of these, use

>>> f = pynbody.load(filename)

to create a :class:`pynbody.snapshot.SimSnap` object *f*, which then acts
as a dictionary holding the arrays inside *f*. For more information
see :ref:`data-access`.

Configuration
-------------

Various aspects of the behaviour of pynbody can be controlled.
See <http://code.google.com/p/pynbody/wiki/ConfigFiles>

Subpackages
-----------

:mod:`~pynbody.array`
    Extends numpy arrays with a custom class array.SimArray
    which holds additional information like units.

:mod:`~pynbody.bridge`
    Allows connections to be made between two different
    SimSnap objects in various ways. 
    <http://code.google.com/p/pynbody/wiki/SameSimulationDifferentOutputs>

:mod:`~pynbody.simdict`
    
:mod:`~pynbody.derived`
    Holds procedures for creating new arrays from existing
    ones, e.g. for getting the radial position. 
    <http://code.google.com/p/pynbody/wiki/AutomagicCalculation>

:mod:`~pynbody.family`
    Stores a registry of different particle types like dm,
    star, gas.
    <http://code.google.com/p/pynbody/wiki/TheFamilySystem>

:mod:`~pynbody.filt`
    Defines and implements 'filters' which allow abstract subsets
    of data to be specified.
    <http://code.google.com/p/pynbody/wiki/FiltersAndSubsims>

:mod:`~pynbody.gadget`
    Implements classes and functions for handling gadget files;
    you rarely need to access this module directly as it will
    be invoked automatically via pynbody.load.

:mod:`~pynbody.gadgethdf`
    Implements classes and functions for handling gadget HDF files
    (if h5py is installed); you rarely need to access this module
    directly as it will be invoked automatically via pynbody.load

:mod:`~pynbody.halo`
    Implements halo catalogue functions. If you have a supported
    halo catalogue on disk or a halo finder installed and
    correctly configured, you can access a halo catalogue through
    f.halos() where f is a SimSnap.
    <http://code.google.com/p/pynbody/wiki/HaloCatalogue>

:mod:`~pynbody.kdtree`
    Implements a KD Tree based on Joachim Stadel's smooth.c.
    You are unlikely to need to access this module directly
    as KD Trees are built in higher level analysis code
    automatically.

:mod:`~pynbody.snapshot`
    Implements the basic SimSnap class and also SubSnap classes
    which can represent different views of the same data.
    <http://code.google.com/p/pynbody/wiki/FiltersAndSubsims>
    You rarely need to access this module directly.

:mod:`~pynbody.sph`
    Allows SPH images to be rendered. The easiest interface
    to this module, at least to start with, is through the
    pynbody.plot package.
    <http://code.google.com/p/pynbody/wiki/SphImages>

:mod:`~pynbody.tipsy`
    Implements classes and functions for handling tipsy files.
    You rarely need to access this module directly as it will
    be invoked automatically via pynbody.load.

:mod:`~pynbody.units`
    Implements a light-weight unit class which is
    used to automatically track units of your simulation arrays.
    <http://code.google.com/p/pynbody/wiki/ConvertingUnits>

:mod:`~pynbody.util`
    Various utility routines used internally by pynbody.

    
"""

from . import backcompat

# Import basic dependencies
import ConfigParser
import os
import imp
import numpy
import warnings
import sys
import logging


# Create config dictionaries which will be required by subpackages
# We use the OrderedDict, which is default in 2.7, but provided here for 2.6/2.5 by
# the backcompat module. This keeps things in the order they were parsed (important
# for units module, for instance).
config_parser = ConfigParser.ConfigParser(dict_type = backcompat.OrderedDict)
config = {}


# Process configuration options
config_parser.optionxform = str
config_parser.read(os.path.join(os.path.dirname(__file__),"default_config.ini"))
config_parser.read(os.path.join(os.path.dirname(__file__),"config.ini"))
config_parser.read(os.path.expanduser("~/.pynbodyrc"))
config_parser.read("config.ini")



config= {'verbose': config_parser.getboolean('general','verbose'),
         'centering-scheme': config_parser.get('general','centering-scheme')}

config['snap-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'snap-class-priority').split(","))
config['halo-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'halo-class-priority').split(","))


config['default-cosmology'] = {}
for k in config_parser.options('default-cosmology') :
    config['default-cosmology'][k] = float(config_parser.get('default-cosmology', k))

config['sph'] = {}
for k in config_parser.options('sph') :
    try:
        config['sph'][k] = int(config_parser.get('sph', k))
    except ValueError:
        pass
    
config['threading'] = config_parser.get('general', 'threading')
config['number_of_threads'] = int(config_parser.get('general', 'number_of_threads'))

config['gravity_calculation_mode'] = config_parser.get('general', 'gravity_calculation_mode')
config['disk-fit-function'] = config_parser.get('general', 'disk-fit-function')

# Create the logger for pynbody
logger = logging.getLogger('pynbody')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s : %(message)s')
for existing_handler in list(logger.handlers) :
    logger.removeHandler(existing_handler)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

if config['verbose'] :
    ch.setLevel(logging.INFO)
    logger.info("Verbose mode is on")
else :
    ch.setLevel(logging.WARNING)
    
    warning = """
Welcome to pynbody v0.30. Note this new version by default is much quieter than old versions.
To get back the verbose output, edit your config.ini or .pynbodyrc file and insert the following
section

[general]
verbose: True

The information is now parsed through python's standard logging module; using logging.getLogger('pynbody')
you can customize the behaviour. See here https://docs.python.org/2/howto/logging-cookbook.html#logging-cookbook."""

    if not os.path.exists(os.path.expanduser("~/.pynbody_v03_touched")) :
        print warning
        with open(os.path.expanduser("~/.pynbody_v03_touched"),"w") as f :
            print>>f, "This file tells pynbody not to reprint the welcome-to-v-0.3 warning"
            

# Import subpackages
from . import util, filt, array, family, snapshot
from .snapshot import tipsy, gadget, gadgethdf, ramses, grafic, nchilada
from . import analysis, halo, derived, bridge, gravity, sph, transformation

try: 
    from . import plot
except: 
    warnings.warn("Unable to import plotting package (missing matplotlib or running from a text-only terminal? Plotting is disabled.", RuntimeWarning)

# The following code resolves inter-dependencies when reloading
imp.reload(array)
imp.reload(util)
# imp.reload(family) # reloading this causes problems for active snapshots
imp.reload(snapshot)
imp.reload(nchilada)
imp.reload(tipsy)
imp.reload(gadget)
imp.reload(gadgethdf)
imp.reload(ramses)
imp.reload(filt)
imp.reload(analysis)
imp.reload(halo)
imp.reload(derived)
imp.reload(bridge)
imp.reload(gravity)
imp.reload(sph)
imp.reload(grafic)

try : 
    imp.reload(plot)
except : 
    pass


# This is our definitive list of classes which are able to
# load snapshots
_snap_classes = [gadgethdf.GadgetHDFSnap, nchilada.NchiladaSnap, gadget.GadgetSnap,
                 tipsy.TipsySnap, ramses.RamsesSnap, grafic.GrafICSnap]

# Turn the config strings for snapshot/halo classes into lists of
# actual classes
_snap_classes_dict = dict([(x.__name__,x) for x in _snap_classes])
_halo_classes_dict = dict([(x.__name__,x) for x in halo._halo_classes])
config['snap-class-priority'] = [_snap_classes_dict[x] for x in config['snap-class-priority']]
config['halo-class-priority'] = [_halo_classes_dict[x] for x in config['halo-class-priority']]


def load(filename, *args, **kwargs) :
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""
   
    for c in config['snap-class-priority'] :
        if c._can_load(filename) :
            logger.info("Loading using backend %s"%str(c))
            return c(filename,*args,**kwargs)

    raise IOError("File %r: format not understood or does not exist"%filename)


from snapshot import _new as new

derived_array = snapshot.SimSnap.derived_quantity

__all__ = ['load', 'new', 'derived_array']
