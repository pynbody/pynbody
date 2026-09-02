"""pytest plugin: route all GadgetHDF/SWIFT array loading through the parallel prototype."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel_proto
from pynbody.snapshot.gadgethdf import HDFArrayLoader

NT = int(os.environ.get("PROTO_THREADS", "4"))
_orig = HDFArrayLoader.load_arrays
def _patched(self, *a):
    return parallel_proto.load_arrays_parallel(self, *a, nthreads=NT)
HDFArrayLoader.load_arrays = _patched
print(f"\n[parallel prototype active, {NT} threads]")
