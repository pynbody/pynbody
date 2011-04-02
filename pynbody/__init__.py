import util, filt, array, family, snapshot,  tipsy, gadget, analysis, halo, derived, bridge, plot

# The following code resolves inter-dependencies when reloading
reload(array)
reload(util)
# reload(family) # reloading this causes problems for active snapshots
reload(snapshot)
reload(tipsy)
reload(gadget)
reload(filt)
reload(analysis)
reload(halo)
reload(derived)
reload(bridge)
reload(plot)

from analysis import profile

_snap_classes = [gadget.GadgetSnap, tipsy.TipsySnap]

def load(filename, *args, **kwargs) :
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""
    for c in _snap_classes :
        if c._can_load(filename) : return c(filename,*args,**kwargs)

    raise RuntimeError("File format not understood")

def new(n_particles=None, **families) :
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled
    with zeros.
    
    By default all particles are taken to be dark matter.
    To specify otherwise, pass in keyword arguments specifying
    the number of particles for each family, e.g.

    f = new(dm=50, star=25, gas=25)
    """

    if len(families)==0 :
        if n_particles is None :
            raise TypeError, "Must specify either the total number of particles or a per-family breakdown"
        families = {'dm': n_particles}

    t_fam = []
    tot_particles = 0


    for k,v in families.items() :
        
        assert isinstance(v,int)
        t_fam.append((family.get_family(k), v))
        tot_particles+=v

    if n_particles is not None :
        assert n_particles==tot_particles
        
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
