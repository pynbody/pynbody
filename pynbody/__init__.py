import ConfigParser, os

config_parser = ConfigParser.ConfigParser()
config_parser.optionxform = str
config_parser.read(os.path.join(os.path.dirname(__file__),"default_config.ini"))
config_parser.read(os.path.join(os.path.dirname(__file__),"config.ini"))



config= {'verbose': config_parser.getboolean('general','verbose'),
         'centering-scheme': config_parser.get('general','centering-scheme')}
config['snap-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'snap-class-priority').split(","))
config['halo-class-priority'] = map(str.strip,
                                    config_parser.get('general', 'halo-class-priority').split(","))




import util, filt, array, family, snapshot,  tipsy, gadget, gadgethdf, analysis, halo, derived, bridge, plot

# The following code resolves inter-dependencies when reloading
reload(array)
reload(util)
# reload(family) # reloading this causes problems for active snapshots
reload(snapshot)
reload(tipsy)
reload(gadget)
reload(gadgethdf)
reload(filt)
reload(analysis)
reload(halo)
reload(derived)
reload(bridge)
reload(plot)

from analysis import profile

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

def new(n_particles=0, **families) :
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled
    with zeros.
    
    By default all particles are taken to be dark matter.
    To specify otherwise, pass in keyword arguments specifying
    the number of particles for each family, e.g.

    f = new(dm=50, star=25, gas=25)
    """

    if len(families)==0 :
        families = {'dm': n_particles}

    t_fam = []
    tot_particles = 0


    for k,v in families.items() :
        
        assert isinstance(v,int)
        t_fam.append((family.get_family(k), v))
        tot_particles+=v

        
    x = snapshot.SimSnap()
    x._num_particles = tot_particles
    x._filename = "<created>"

    x._create_arrays(["pos","vel"],3)
    x._create_arrays(["mass"],1)
    
    rt = 0
    for k,v in t_fam :
        x._family_slice[k] = slice(rt,rt+v)
        rt+=v

    x._decorate()
    return x
