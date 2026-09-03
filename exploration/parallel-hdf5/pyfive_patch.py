"""pytest plugin: make pynbody's GadgetHDF reader open files with pyfive instead of h5py.

Includes the two trivial API shims pyfive is missing relative to h5py, so that the
failures that remain are real coverage gaps rather than cosmetic ones.
"""
import pyfive
from pynbody.snapshot import gadgethdf

if not hasattr(pyfive.high_level.Dataset, "__len__"):
    pyfive.high_level.Dataset.__len__ = lambda self: self.shape[0]

def _open(filename, mode='r'):
    if mode != 'r':
        raise NotImplementedError("pyfive is read-only")
    return pyfive.File(str(filename))

gadgethdf._open_hdf_file = _open
print("\n[pyfive substituted for h5py in gadgethdf._open_hdf_file, with __len__ shim]")
