import pynbody.snapshot, pynbody.tipsy

# The following code resolves inter-dependencies when reloading
reload(pynbody.array)
reload(pynbody.util)
reload(pynbody.snapshot)
reload(pynbody.tipsy)
reload(pynbody.gadget)
reload(pynbody.snapshot)


from pynbody.snapshot import load

