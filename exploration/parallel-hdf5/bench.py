import os, sys, time, subprocess, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pynbody
from pynbody.snapshot import gadgethdf
import parallel_proto
S = os.path.dirname(os.path.abspath(__file__))

def cold():
    subprocess.run("sync", shell=True)
    open("/proc/sys/vm/drop_caches","w").write("3\n")

def run(path, nthreads, take=None):
    orig = gadgethdf.HDFArrayLoader.load_arrays
    if nthreads:
        gadgethdf.HDFArrayLoader.load_arrays = (
            lambda self, *a: parallel_proto.load_arrays_parallel(self, *a, nthreads=nthreads))
    try:
        cold()
        t0=time.perf_counter()
        f = pynbody.load(path, **({} if take is None else dict(take=take)))
        for k in ("pos","vel","iord","mass"): f.dm[k]
        dt=time.perf_counter()-t0
        nb = sum(f.dm[k].nbytes for k in ("pos","vel","iord","mass"))
        return dt, nb
    finally:
        gadgethdf.HDFArrayLoader.load_arrays = orig

for label, path in (("uncompressed", f"{S}/snap/snap"), ("gzip+shuffle", f"{S}/snapgz/snap")):
    print(f"\n{label}:")
    base=None
    for nt in (0, 2, 4, 8):
        dt, nb = run(path, nt)
        if base is None: base = dt
        tag = "serial (current code)" if nt==0 else f"parallel, {nt} threads"
        print(f"  {tag:28s} {dt:6.3f} s   {nb/1e9/dt:5.2f} GB/s   speedup x{base/dt:4.2f}")

print("\npartial load (take = every 7th particle), uncompressed:")
take = np.arange(0, 16000000, 7)
base=None
for nt in (0, 4, 8):
    dt, nb = run(f"{S}/snap/snap", nt, take=take)
    if base is None: base=dt
    tag = "serial (current code)" if nt==0 else f"parallel, {nt} threads"
    print(f"  {tag:28s} {dt:6.3f} s   {nb/1e9/dt:5.2f} GB/s   speedup x{base/dt:4.2f}")
