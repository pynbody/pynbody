import util, array, decorate, family, snapshot,  tipsy, gadget

# The following code resolves inter-dependencies when reloading
reload(array)
reload(util)
reload(decorate)
# reload(family) # reloading this causes problems for active snapshots
reload(snapshot)
reload(tipsy)
reload(gadget)

_snap_classes = [gadget.GadgetSnap, tipsy.TipsySnap]

def load(filename, *args) :
    """Loads a file using the appropriate class, returning a SimSnap
    instance."""
    for c in _snap_classes :
	if c._can_load(filename) : return c(filename,*args)
	
    raise RuntimeError("File format not understood")



