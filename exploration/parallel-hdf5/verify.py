"""Check the parallel prototype reproduces the serial loader byte-for-byte."""
import os, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, numpy.testing as npt, pynbody
from pynbody.snapshot import gadgethdf
import parallel_proto

S = os.path.dirname(os.path.abspath(__file__))
PATH = os.environ.get("SNAP", f"{S}/snap/snap")

def load(path, parallel, nthreads=4, **kw):
    orig = gadgethdf.HDFArrayLoader.load_arrays
    if parallel:
        gadgethdf.HDFArrayLoader.load_arrays = (
            lambda self, *a: parallel_proto.load_arrays_parallel(self, *a, nthreads=nthreads))
    try:
        f = pynbody.load(path, **kw)
        out = {k: np.array(f.dm[k]) for k in ("pos","vel","iord","mass")}
        return out, len(f)
    finally:
        gadgethdf.HDFArrayLoader.load_arrays = orig

cases = [
    ("full load", {}),
    ("take = every 7th particle", dict(take=np.arange(0, 16000000, 7))),
    ("take = scattered block", dict(take=np.concatenate([
        np.arange(1999000, 2001000), np.arange(5_000_000, 5_000_050),
        np.arange(15_999_000, 16_000_000)]))),
    ("take = slice across file boundary", dict(take=np.arange(1_999_990, 2_000_010))),
    ("take = single particle", dict(take=np.array([12345678]))),
    ("take = starts at chunk boundary", dict(take=np.arange(524288, 524288+10))),
    ("take = empty", dict(take=np.array([], dtype=np.int64))),
]
for label, kw in cases:
    a, na = load(PATH, False, **kw)
    b, nb = load(PATH, True, **kw)
    assert na == nb, (na, nb)
    for k in a:
        npt.assert_array_equal(a[k], b[k], err_msg=f"{label}: {k} differs")
    print(f"  OK  {label:38s} ({na} particles)")
print("all cases match")
