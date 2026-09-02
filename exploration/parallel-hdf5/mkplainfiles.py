import h5py, numpy as np, os, sys
d = os.path.dirname(os.path.abspath(__file__))
N = 8*1024*1024   # 8M float64 = 64 MB per file
rng = np.random.default_rng(1)
for i in range(8):
    a = rng.standard_normal(N)
    with h5py.File(f"{d}/plain.{i}.h5","w") as f:
        f.create_dataset("x", data=a)
    with h5py.File(f"{d}/gz.{i}.h5","w") as f:
        f.create_dataset("x", data=a, chunks=(1024*1024,), compression="gzip", compression_opts=4)
    print("wrote", i, flush=True)
