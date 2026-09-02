"""Separate the causes: (a) processes vs threads, (b) is `phil` the serialiser?"""
import os, time, h5py, numpy as np, contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
d = os.path.dirname(os.path.abspath(__file__))
N = 8*1024*1024; NF = 8

def read_one(args):
    kind, i = args
    buf = np.empty(N)
    with h5py.File(f"{d}/{kind}.{i}.h5","r") as f:
        f["x"].read_direct(buf)
    return float(buf[0])

def timeit(f):
    t0=time.perf_counter(); c0=time.process_time(); f()
    return time.perf_counter()-t0, time.process_time()-c0

if __name__ == "__main__":
    # warm cache
    for i in range(NF):
        for k in ("plain","gz"):
            with open(f"{d}/{k}.{i}.h5","rb") as fh:
                while fh.read(1<<22): pass

    args = lambda kind: [(kind,i) for i in range(NF)]
    for kind in ("plain","gz"):
        seq,_  = timeit(lambda: [read_one(a) for a in args(kind)])
        with ThreadPoolExecutor(4) as ex:
            thr,cpu = timeit(lambda: list(ex.map(read_one, args(kind))))
        with ProcessPoolExecutor(4) as ex:
            ex.map(read_one, [(kind,0)])  # warm workers
            prc,_ = timeit(lambda: list(ex.map(read_one, args(kind))))
        print(f"{kind:6s} seq={seq:6.3f}s  4-thread={thr:6.3f}s (x{seq/thr:.2f})  4-process={prc:6.3f}s (x{seq/prc:.2f})")

    # Is phil the serialiser?  Swap it for a no-op and see if threads scale.
    import h5py._objects as ho
    class NoLock:
        def __enter__(self): return True
        def __exit__(self,*a): return False
        def acquire(self,*a,**k): return True
        def release(self): pass
    orig = ho.phil
    try:
        ho.phil = NoLock()
        ok = isinstance(ho.phil, NoLock)
    except Exception as e:
        ok = False; print("cannot rebind phil:", e)
    print("phil rebound in module namespace:", ok)
    if ok:
        for kind in ("plain","gz"):
            with ThreadPoolExecutor(4) as ex:
                thr,cpu = timeit(lambda: list(ex.map(read_one, args(kind))))
            print(f"  [phil disabled] {kind:6s} 4-thread={thr:6.3f}s cpu={cpu:6.3f}s")
    ho.phil = orig
