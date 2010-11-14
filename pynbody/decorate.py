"""Define a system for 'decorating' simulation snapshots after
they have been loaded in. Decorating will often involve looking
for and setting auxiliary information e.g. on units for arrays,
or cosmological information like omegaM0 etc.

Most people will never need to look at this system, but to define your
own decorator which will be automatically called after any snapshot is
loaded in, declare a function as follows:

@pynbody.decorate.sim_decorator
def my_decorate(sim) :
    # Called whenever a new snapshot is created
    # Do work on Snapshot object sim here

"""

_registry = []
_sub_registry = []

def sim_decorator(fn) :
    """A decorator to mark a function as a simulation decorator, i.e.
    to be called whenever a top-level SnapShot is instantiated. """

    _registry.append(fn)
    return fn

def sub_decorator(fn) :
    """A decorator to mark a function as a sub-simulation decorator,
    i.e. to be called whenever a SubSnap is instantiated."""
    _sub_registry.append(fn)
    return fn

def decorate_top_snap(sim) :
    """Called by relevant SimSnap child classes when initialization of
    a top-level (i.e. no base) SimSnap is complete to perform
    decoration of the new instance."""
    for fn in _registry :
	fn(sim)

def decorate_sub_snap(sim) :
    """Called when initialization of a SubSnap (or equivalent)
    is complete, to perform decoration of the new instance."""
    for fn in _registry :
	fn(sim)

