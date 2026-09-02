"""Compare candidate parallel-loading strategies, always from a cold page cache.

NB: this sandbox's virtio block device is not a Lustre client; the point here is
to show whether/how much extra *concurrency* (queue depth) changes throughput,
not to predict absolute numbers.
"""
import os, sys, time, subprocess, numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
S = os.path.dirname(os.path.abspath(__file__))
files = [f"{S}/snap/snap.{i}.hdf5" for i in range(8)]
KEY = "PartType1/Coordinates"
NPART = 2000000; NTOT = NPART*len(files)

def cold():
    subprocess.run("sync", shell=True)
    with open("/proc/sys/vm/drop_caches","w") as f: f.write("3\n")

def layout():
    out=[]
    for fn in files:
        with h5py.File(fn,'r') as f:
            d=f[KEY]
            out.append(dict(fn=fn, off=d.id.get_offset(), nbytes=d.nbytes,
                            shape=d.shape, dtype=d.dtype))
    return out
LAY = layout()

def serial_h5py(big):
    off=0
    for fn in files:
        with h5py.File(fn,'r') as f:
            d=f[KEY]; d.read_direct(big[off:off+d.shape[0]]); off+=d.shape[0]

def threads_pread(big, nt):
    def one(i):
        L=LAY[i]; off=i*NPART
        mv = memoryview(big[off:off+NPART]).cast('B')
        fd = os.open(L['fn'], os.O_RDONLY)
        try:
            done=0
            while done < L['nbytes']:
                n=os.preadv(fd, [mv[done:]], L['off']+done)
                if n==0: raise IOError
                done+=n
        finally: os.close(fd)
    with ThreadPoolExecutor(nt) as ex: list(ex.map(one, range(len(files))))

def serial_h5py_with_prefetch(big, nt):
    """h5py stays serial (phil), but N threads pull the bytes into page cache ahead of it."""
    scratch = [bytearray(1<<22) for _ in range(nt)]
    def prefetch(i):
        L=LAY[i]; fd=os.open(L['fn'], os.O_RDONLY)
        buf = scratch[i % nt]; mv=memoryview(buf)
        try:
            done=0
            while done < L['nbytes']:
                n=os.preadv(fd,[mv],L['off']+done)
                if n==0: break
                done+=n
        finally: os.close(fd)
    ex = ThreadPoolExecutor(nt)
    futs=[ex.submit(prefetch,i) for i in range(len(files))]
    serial_h5py(big)
    ex.shutdown(wait=True)

def _proc_read(i):
    L=LAY[i]
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    arr = np.ndarray((NTOT,3), np.float32, buffer=shm.buf)
    with h5py.File(L['fn'],'r') as f:
        f[KEY].read_direct(arr[i*NPART:(i+1)*NPART])
    shm.close()
    return i

def timeit(label, fn):
    cold(); t0=time.perf_counter(); fn(); dt=time.perf_counter()-t0
    gb = NTOT*3*4/1e9
    print(f"{label:44s} {dt:6.3f} s   {gb/dt:6.2f} GB/s")
    return dt

if __name__ == "__main__":
    from multiprocessing import shared_memory
    print(f"{NTOT/1e6:.0f}M particles, {NTOT*3*4/1e9:.2f} GB of Coordinates across {len(files)} files\n")
    for rep in range(2):
        big = np.empty((NTOT,3), np.float32)
        timeit("serial h5py (status quo)", lambda: serial_h5py(big))
        for nt in (2,4,8):
            timeit(f"raw preadv, {nt} threads", lambda nt=nt: threads_pread(big, nt))
        for nt in (4,8):
            timeit(f"serial h5py + {nt} prefetch threads", lambda nt=nt: serial_h5py_with_prefetch(big, nt))
        shm = shared_memory.SharedMemory(create=True, size=NTOT*3*4)
        SHM_NAME = shm.name
        globals()['SHM_NAME'] = shm.name
        for nt in (4,8):
            with ProcessPoolExecutor(nt, mp_context=__import__('multiprocessing').get_context('fork')) as ex:
                ex.submit(int, 1).result()
                timeit(f"h5py in {nt} processes -> shared memory", lambda ex=ex: list(ex.map(_proc_read, range(len(files)))))
        shm.close(); shm.unlink()
        print()
