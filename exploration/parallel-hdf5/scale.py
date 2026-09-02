"""Does concurrent h5py reading from separate files actually scale?"""
import os, sys, time, threading, h5py, numpy as np
from concurrent.futures import ThreadPoolExecutor
d = os.path.dirname(os.path.abspath(__file__))
N = 8*1024*1024
NF = 8

def warm():
    for i in range(NF):
        for k in ("plain","gz"):
            with open(f"{d}/{k}.{i}.h5","rb") as f:
                while f.read(1<<22): pass

def read_one(kind, i, buf):
    with h5py.File(f"{d}/{kind}.{i}.h5","r") as f:
        f["x"].read_direct(buf)
    return buf[0]

def run(kind, nthreads):
    bufs = [np.empty(N) for _ in range(NF)]
    t0 = time.perf_counter(); c0 = time.process_time()
    if nthreads == 1:
        for i in range(NF): read_one(kind, i, bufs[i])
    else:
        with ThreadPoolExecutor(nthreads) as ex:
            list(ex.map(lambda i: read_one(kind, i, bufs[i]), range(NF)))
    dt = time.perf_counter()-t0; dc = time.process_time()-c0
    return dt, dc

warm()
print(f"{'kind':6s} {'thr':>4s} {'wall(s)':>9s} {'cpu(s)':>8s} {'GB/s':>7s} {'speedup':>8s}")
for kind in ("plain","gz"):
    base = None
    for nt in (1,2,4,8):
        best = min(run(kind, nt)[0] for _ in range(3))
        dt, dc = run(kind, nt)
        dt = min(dt, best)
        if base is None: base = dt
        gb = NF*N*8/1e9
        print(f"{kind:6s} {nt:4d} {dt:9.3f} {dc:8.3f} {gb/dt:7.2f} {base/dt:8.2f}")
