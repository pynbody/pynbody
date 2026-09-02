import os, sys, time, subprocess, statistics, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pynbody
from pynbody.snapshot import gadgethdf
import parallel_proto
S = os.path.dirname(os.path.abspath(__file__))
def cold():
    subprocess.run("sync", shell=True); open("/proc/sys/vm/drop_caches","w").write("3\n")
def run(path, nthreads):
    orig = gadgethdf.HDFArrayLoader.load_arrays
    if nthreads:
        gadgethdf.HDFArrayLoader.load_arrays = (
            lambda self,*a: parallel_proto.load_arrays_parallel(self,*a,nthreads=nthreads))
    try:
        cold(); t0=time.perf_counter()
        f=pynbody.load(path)
        for k in ("pos","vel","iord","mass"): f.dm[k]
        return time.perf_counter()-t0
    finally:
        gadgethdf.HDFArrayLoader.load_arrays = orig
for label, path in (("uncompressed", f"{S}/snap/snap"), ("gzip+shuffle", f"{S}/snapgz/snap")):
    print(f"\n{label}  (3 cold repeats, median):")
    med={}
    for nt in (0,4,8):
        ts=[run(path,nt) for _ in range(3)]
        med[nt]=statistics.median(ts)
        tag="serial" if nt==0 else f"{nt} threads"
        print(f"  {tag:10s} {med[nt]:6.3f} s   (runs {['%.2f'%t for t in ts]})  x{med[0]/med[nt]:.2f}")
