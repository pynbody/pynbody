"""Confirm h5py's global lock (phil) is what blocks concurrency, and that raw
pread on the dataset's byte offset escapes it."""
import os, time, threading, h5py, numpy as np
from concurrent.futures import ThreadPoolExecutor
d = os.path.dirname(os.path.abspath(__file__))
N = 8*1024*1024; NF = 8

for i in range(NF):
    with open(f"{d}/plain.{i}.h5","rb") as fh:
        while fh.read(1<<22): pass

# --- 1. does a long read in thread A block a *trivial* h5py call in thread B?
blocked = []
def slow_read():
    buf = np.empty(N)
    with h5py.File(f"{d}/plain.0.h5","r") as f:
        for _ in range(6): f["x"].read_direct(buf)
def tiny_call():
    time.sleep(0.05)
    f = h5py.File(f"{d}/plain.1.h5","r")
    t0 = time.perf_counter(); _ = f["x"].shape; blocked.append(time.perf_counter()-t0)
    f.close()
tA = threading.Thread(target=slow_read); tB = threading.Thread(target=tiny_call)
t0=time.perf_counter(); tA.start(); tB.start(); tA.join(); tB.join()
print(f"long read took {time.perf_counter()-t0:.3f}s; trivial h5py call in other thread blocked for {blocked[0]*1e3:.1f} ms")

# --- 2. contiguous dataset byte offsets, and raw pread scaling
offs = []
for i in range(NF):
    with h5py.File(f"{d}/plain.{i}.h5","r") as f:
        dsid = f["x"].id
        offs.append((dsid.get_offset(), f["x"].nbytes, f["x"].id.get_storage_size(), f["x"].chunks))
print("dataset offset/nbytes/storage/chunks for file 0:", offs[0])

fds = [os.open(f"{d}/plain.{i}.h5", os.O_RDONLY) for i in range(NF)]
bufs = [np.empty(N) for _ in range(NF)]
def pread_one(i):
    off, nbytes = offs[i][0], offs[i][1]
    mv = memoryview(bufs[i]).cast('B')
    done = 0
    while done < nbytes:
        n = os.preadv(fds[i], [mv[done:]], off+done)
        if n == 0: raise IOError
        done += n
    return done

def timeit(fn):
    t0=time.perf_counter(); c0=time.process_time(); fn()
    return time.perf_counter()-t0, time.process_time()-c0

seq,_ = timeit(lambda: [pread_one(i) for i in range(NF)])
for nt in (2,4,8):
    with ThreadPoolExecutor(nt) as ex:
        w,c = timeit(lambda: list(ex.map(pread_one, range(NF))))
    print(f"raw preadv  {nt} threads: wall={w:.3f}s cpu={c:.3f}s  speedup vs seq({seq:.3f}s) = x{seq/w:.2f}")

# sanity: same data?
b = np.empty(N)
with h5py.File(f"{d}/plain.3.h5","r") as f: f["x"].read_direct(b)
print("raw pread matches h5py read:", np.array_equal(b, bufs[3]))
for fd in fds: os.close(fd)
