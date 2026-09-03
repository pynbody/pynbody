"""Do the non-h5py HDF5 readers actually scale across threads?"""
import os, sys, time, subprocess, statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor
S = os.path.dirname(os.path.abspath(__file__))
D = f"{S}/h5test"
NF = 8

def cold():
    subprocess.run("sync", shell=True)
    try: open("/proc/sys/vm/drop_caches","w").write("3\n")
    except Exception: pass

def warm():
    for k in ("plain","gz"):
        for i in range(NF):
            with open(f"{D}/{k}.{i}.h5","rb") as f:
                while f.read(1<<22): pass

def timeit(fn):
    t0=time.perf_counter(); c0=time.process_time(); fn()
    return time.perf_counter()-t0, time.process_time()-c0

import h5py
def read_h5py(kind, i):
    with h5py.File(f"{D}/{kind}.{i}.h5","r") as f:
        return f["x"][:].sum()

import pyfive
def read_pyfive(kind, i):
    with pyfive.File(f"{D}/{kind}.{i}.h5") as f:
        return f["x"][:].sum()

readers = {"h5py": read_h5py, "pyfive": read_pyfive}
try:
    import hidefix
    # hidefix separates indexing from reading; index once up front, as intended
    _idx = {(k,i): hidefix.Index(f"{D}/{k}.{i}.h5") for k in ("plain","gz") for i in range(NF)}
    def read_hidefix(kind, i):
        return _idx[(kind,i)].dataset("x")[(slice(None),)].sum()
    readers["hidefix"] = read_hidefix
except Exception as e:
    print(f"(hidefix unavailable: {type(e).__name__}: {e})")

warm()
print(f"{'reader':9s} {'kind':6s} {'threads':>7s} {'wall':>8s} {'cpu':>8s} {'GB/s':>7s} {'speedup':>8s}   cores busy")
for name, rd in readers.items():
    for kind in ("plain","gz"):
        base=None
        for nt in (1,2,4):
            def run(nt=nt, rd=rd, kind=kind):
                if nt==1:
                    for i in range(NF): rd(kind,i)
                else:
                    with ThreadPoolExecutor(nt) as ex:
                        list(ex.map(lambda i: rd(kind,i), range(NF)))
            w,c = min((timeit(run) for _ in range(2)), key=lambda t: t[0])
            if base is None: base=w
            gb = NF*8*1024*1024*8/1e9
            print(f"{name:9s} {kind:6s} {nt:7d} {w:7.3f}s {c:7.3f}s {gb/w:7.2f} {base/w:8.2f}   {c/w:5.2f}")
    print()
