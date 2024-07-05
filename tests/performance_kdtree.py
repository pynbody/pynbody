import contextlib
import sys
import time

import numpy as np

import pynbody


@contextlib.contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end-start:.2f}s")

print("""performance_kdtree.py

This script is designed to test the performance of the KDTree.
It does not test the correctness, for which the normal unit tests should be used.

You can test with different numbers of threads by passing the number of threads as an argument to this script.

""")

try:
    num_threads = int(sys.argv[1])
    print("Using", num_threads, "threads for KDTree operations")
except Exception:
    num_threads = None

np.random.seed(1337)

Npart = 10000000
Ntrials = 20
dtype = np.float32

centres = (np.random.uniform(size=(Ntrials,3)) - 0.5).astype(dtype)
radii = 10.**np.random.uniform(low=-3.0,high=-1.0,size=Ntrials).astype(dtype)

xcs = np.random.uniform(-0.5, 0.5, size=Ntrials).astype(dtype)
ycs = np.random.uniform(-0.5, 0.5, size=Ntrials).astype(dtype)
zcs = np.random.uniform(-0.5, 0.5, size=Ntrials).astype(dtype)

snap = pynbody.new(dm=Npart)
snap._create_array('pos', 3, dtype=dtype)
snap._create_array('mass', 1, dtype=dtype)

snap['pos'] = np.random.uniform(size=(Npart, 3)).astype(dtype)
snap['pos'] -= 0.5
snap['mass'] = np.ones(Npart, dtype=dtype)
snap.properties['boxsize'] = 1.0

print("Using data type = ", snap['pos'].dtype)

with timer("sphere queries without tree"):
    for cen, rad in zip(centres, radii):
        print(".",end="")
        sys.stdout.flush()
        _ = snap[pynbody.filt.Sphere(rad, cen)]

with timer("cube queries without tree"):
    for xc, yc, zc, s in zip(xcs, ycs, zcs, radii):
        print(".",end="")
        sys.stdout.flush()
        _ = snap[pynbody.filt.Cuboid(xc - s, yc - s, zc - s, xc + s, yc + s, zc + s)]


with timer("tree build"):
    snap.build_tree(num_threads=num_threads)



with timer("sphere queries from tree"):
    for cen,rad in zip(centres,radii):
        print(".", end="")
        sys.stdout.flush()
        _ = snap[pynbody.filt.Sphere(rad, cen)]

# with timer("cube queries from tree"):
#     for xc, yc, zc, s in zip(xcs, ycs, zcs, radii):
#         print(".", end="")
#         sys.stdout.flush()
#         _ = f[pynbody.filt.Cuboid(xc - s, yc - s, zc - s, xc + s, yc + s, zc + s)]



with timer("get smooth"):
    _ = snap['smooth']

with timer("get rho"):
    _ = snap['rho']
