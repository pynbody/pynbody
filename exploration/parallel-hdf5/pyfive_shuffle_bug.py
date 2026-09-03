"""Reproducer: pyfive misreads a chunk whose filter pipeline puts fletcher32 first.

pyfive assumes h5py's default filter order (shuffle -> deflate -> fletcher32). SWIFT
writes fletcher32 -> shuffle -> deflate, so on read the 4-byte checksum is still
attached when pyfive unshuffles: a 4100-byte buffer at itemsize 8 gives 513 elements
instead of 512.

This affects SWIFT's Cells/Counts datasets, which is what take_region= reads.
"""
import os

import h5py
import numpy as np
import pyfive

PATH = "pyfive_filter_order.h5"
a = np.arange(512, dtype="<i8")

# build the pipeline in SWIFT's order via the low-level API; h5py's create_dataset
# keyword arguments always emit shuffle first, which pyfive handles correctly
dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
dcpl.set_chunk((512,))
dcpl.set_fletcher32()
dcpl.set_shuffle()
dcpl.set_deflate(4)

fid = h5py.h5f.create(PATH.encode(), h5py.h5f.ACC_TRUNC)
dsid = h5py.h5d.create(fid, b"d", h5py.h5t.py_create(a.dtype, logical=True),
                       h5py.h5s.create_simple((512,)), dcpl=dcpl)
dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, a)
del dsid
fid.close()

with h5py.File(PATH, "r") as f:
    plist = f["d"].id.get_create_plist()
    print("filter order:", [plist.get_filter(i)[0] for i in range(plist.get_nfilters())],
          "(3=fletcher32, 2=shuffle, 1=deflate)")
    print("h5py  :", "OK" if np.array_equal(a, f["d"][...]) else "WRONG")

try:
    with pyfive.File(PATH) as f:
        print("pyfive:", "OK" if np.array_equal(a, f["d"][...]) else "WRONG")
except Exception as e:
    print(f"pyfive: {type(e).__name__}: {e}")

os.unlink(PATH)
