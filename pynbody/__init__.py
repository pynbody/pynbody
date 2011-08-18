# Import basic dependencies
import ConfigParser
import os
import imp
import numpy

# Create config dictionaries which will be required by subpackages
config_parser = ConfigParser.ConfigParser()
config = {}


# Process configuration options
config_parser.optionxform = str
config_parser.read(os.path.join(os.path.dirname(__file__),"default_config.ini"))
config_parser.read(os.path.join(os.path.dirname(__file__),"config.ini"))
config_parser.read(os.path.expanduser("~/.pynbodyrc"))
config_parser.read("config.ini")


config.update({'verbose': config_parser.getboolean('general','verbose'),
         'centering-scheme': config_parser.get('general','centering-scheme')})

config['snap-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'snap-class-priority').split(","))
config['halo-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'halo-class-priority').split(","))


config['default-cosmology'] = {}
for k in config_parser.options('default-cosmology') :
    config['default-cosmology'][k] = float(config_parser.get('default-cosmology', k))


# Import subpackages
from . import util, filt, array, family, snapshot,  tipsy, gadget, gadgethdf, analysis, halo, derived, bridge, plot

# The following code resolves inter-dependencies when reloading
imp.reload(array)
imp.reload(util)
# imp.reload(family) # reloading this causes problems for active snapshots
imp.reload(snapshot)
imp.reload(tipsy)
imp.reload(gadget)
imp.reload(gadgethdf)
imp.reload(filt)
imp.reload(analysis)
imp.reload(halo)
imp.reload(derived)
imp.reload(bridge)
imp.reload(plot)

# from analysis import profile


# This is our definitive list of classes which are able to
# load snapshots
_snap_classes = [gadgethdf.GadgetHDFSnap, gadget.GadgetSnap, tipsy.TipsySnap]


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
            if config['verbose'] : print "Attempting to load as",c
            return c(filename,*args,**kwargs)

    raise RuntimeError("File format not understood")


from snapshot import _new as new
