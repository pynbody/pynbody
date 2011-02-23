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
