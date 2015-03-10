"""
pynbody
=======

A light-weight, portable, format-transparent analysis framework
for N-body and SPH astrophysical simulations.

For more information, either build the latest documentation included
in our git repository, or view the online version here:
http://pynbody.github.io/pynbody/

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
import multiprocessing


# Create config dictionaries which will be required by subpackages
# We use the OrderedDict, which is default in 2.7, but provided here for 2.6/2.5 by
# the backcompat module. This keeps things in the order they were parsed (important
# for units module, for instance).
config_parser = ConfigParser.ConfigParser(dict_type=backcompat.OrderedDict)
config = {}


# Process configuration options
config_parser.optionxform = str
config_parser.read(
    os.path.join(os.path.dirname(__file__), "default_config.ini"))
config_parser.read(os.path.join(os.path.dirname(__file__), "config.ini"))
config_parser.read(os.path.expanduser("~/.pynbodyrc"))
config_parser.read("config.ini")


config = {'verbose': config_parser.getboolean('general', 'verbose'),
          'centering-scheme': config_parser.get('general', 'centering-scheme')}

config['snap-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'snap-class-priority').split(","))
config['halo-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'halo-class-priority').split(","))


config['default-cosmology'] = {}
for k in config_parser.options('default-cosmology'):
    config[
        'default-cosmology'][k] = float(config_parser.get('default-cosmology', k))

config['sph'] = {}
for k in config_parser.options('sph'):
    try:
        config['sph'][k] = int(config_parser.get('sph', k))
    except ValueError:
        pass

config['threading'] = config_parser.get('general', 'threading')
config['number_of_threads'] = int(
    config_parser.get('general', 'number_of_threads'))

if config['number_of_threads']<0:
    config['number_of_threads']=multiprocessing.cpu_count()

config['gravity_calculation_mode'] = config_parser.get(
    'general', 'gravity_calculation_mode')
config['disk-fit-function'] = config_parser.get('general', 'disk-fit-function')

# Create the logger for pynbody
logger = logging.getLogger('pynbody')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s : %(message)s')
for existing_handler in list(logger.handlers):
    logger.removeHandler(existing_handler)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

if config['verbose']:
    ch.setLevel(logging.INFO)
    logger.info("Verbose mode is on")
else:
    ch.setLevel(logging.WARNING)

    warning = """
Welcome to pynbody v0.30. Note this new version by default is much quieter than old versions.
To get back the verbose output, edit your config.ini or .pynbodyrc file and insert the following
section

[general]
verbose: True

The information is now parsed through python's standard logging module; using logging.getLogger('pynbody')
you can customize the behaviour. See here https://docs.python.org/2/howto/logging-cookbook.html#logging-cookbook."""

    if not os.path.exists(os.path.expanduser("~/.pynbody_v03_touched")):
        print warning
        with open(os.path.expanduser("~/.pynbody_v03_touched"), "w") as f:
            print>>f, "This file tells pynbody not to reprint the welcome-to-v-0.3 warning"


# Import subpackages
from . import util, filt, array, family, snapshot
from .snapshot import tipsy, gadget, gadgethdf, ramses, grafic, nchilada, ascii
from . import analysis, halo, derived, bridge, gravity, sph, transformation

try:
    from . import plot
except:
    warnings.warn(
        "Unable to import plotting package (missing matplotlib or running from a text-only terminal? Plotting is disabled.", RuntimeWarning)

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

try:
    imp.reload(plot)
except:
    pass


# This is our definitive list of classes which are able to
# load snapshots
_snap_classes = [gadgethdf.GadgetHDFSnap, nchilada.NchiladaSnap, gadget.GadgetSnap,
                 tipsy.TipsySnap, ramses.RamsesSnap, grafic.GrafICSnap,
                 ascii.AsciiSnap]

# Turn the config strings for snapshot/halo classes into lists of
# actual classes
_snap_classes_dict = dict([(x.__name__, x) for x in _snap_classes])
_halo_classes_dict = dict([(x.__name__, x) for x in halo._halo_classes])
config['snap-class-priority'] = [_snap_classes_dict[x]
                                 for x in config['snap-class-priority']]
config['halo-class-priority'] = [_halo_classes_dict[x]
                                 for x in config['halo-class-priority']]


def load(filename, *args, **kwargs):
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""

    for c in config['snap-class-priority']:
        if c._can_load(filename):
            logger.info("Loading using backend %s" % str(c))
            return c(filename, *args, **kwargs)

    raise IOError(
        "File %r: format not understood or does not exist" % filename)


from snapshot import _new as new

derived_array = snapshot.SimSnap.derived_quantity

__all__ = ['load', 'new', 'derived_array']
