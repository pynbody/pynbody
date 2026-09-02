"""Detect whether h5py releases the GIL during a read.

A monitor thread polls a monotonic clock in a tight pure-Python loop and records
the largest gap between consecutive polls.  If the reading thread holds the GIL
for the whole of an HDF5 read, the monitor cannot run at all for that period, so
max_gap ~ read duration.  If the GIL is released, max_gap stays at the
sys.setswitchinterval() scale (~5 ms).
"""
import sys, threading, time, h5py, numpy as np, os
d = os.path.dirname(os.path.abspath(__file__))
sys.setswitchinterval(0.001)

def monitor(stop, out):
    mx = 0.0; last = time.perf_counter(); n = 0
    while not stop.is_set():
        now = time.perf_counter()
        gap = now - last
        if gap > mx: mx = gap
        last = now; n += 1
    out.append((mx, n))

def probe(label, fn):
    stop = threading.Event(); out = []
    t = threading.Thread(target=monitor, args=(stop, out)); t.start()
    time.sleep(0.2)
    t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
    stop.set(); t.join()
    mx, n = out[0]
    print(f"{label:38s} read={dt*1e3:8.1f} ms   max GIL stall={mx*1e3:8.1f} ms  polls={n}")
    return dt, mx

# baseline: a numpy op known to release the GIL, and one that does not (pure python)
a = np.empty(8*1024*1024)
probe("numpy sqrt (releases GIL? partially)", lambda: np.sqrt(a, out=a))

def read_plain_getitem():
    with h5py.File(f"{d}/plain.0.h5","r") as f:
        f["x"][:]
def read_plain_direct():
    buf = np.empty(8*1024*1024)
    with h5py.File(f"{d}/plain.0.h5","r") as f:
        f["x"].read_direct(buf)
def read_gz_getitem():
    with h5py.File(f"{d}/gz.0.h5","r") as f:
        f["x"][:]
def read_gz_direct():
    buf = np.empty(8*1024*1024)
    with h5py.File(f"{d}/gz.0.h5","r") as f:
        f["x"].read_direct(buf)

for _ in range(2):
    probe("h5py plain dset[:]", read_plain_getitem)
    probe("h5py plain read_direct", read_plain_direct)
    probe("h5py gzip dset[:]", read_gz_getitem)
    probe("h5py gzip read_direct", read_gz_direct)
    print()
