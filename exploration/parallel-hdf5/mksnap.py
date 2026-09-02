"""Write a synthetic multi-file Gadget-HDF5 snapshot for profiling pynbody's loader."""
import h5py, numpy as np, os, sys

outdir = sys.argv[1]; NFILES = int(sys.argv[2]); NPART = int(sys.argv[3])
compress = (len(sys.argv) > 4 and sys.argv[4] == "gzip")
os.makedirs(outdir, exist_ok=True)
kw = dict(compression="gzip", compression_opts=4, shuffle=True,
          chunks=(min(NPART, 1<<20),)) if compress else {}
kw3 = dict(compression="gzip", compression_opts=4, shuffle=True,
           chunks=(min(NPART, 1<<19), 3)) if compress else {}

rng = np.random.default_rng(42)
for i in range(NFILES):
    with h5py.File(f"{outdir}/snap.{i}.hdf5", "w") as f:
        h = f.create_group("Header")
        h.attrs["NumFilesPerSnapshot"] = NFILES
        h.attrs["NumPart_ThisFile"] = np.array([0, NPART, 0, 0, 0, 0], dtype=np.int32)
        h.attrs["NumPart_Total"] = np.array([0, NPART*NFILES, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Redshift"] = 0.0
        h.attrs["Time"] = 1.0
        h.attrs["Time_GYR"] = 13.8
        h.attrs["BoxSize"] = 100000.0
        h.attrs["Omega0"] = 0.3
        h.attrs["OmegaLambda"] = 0.7
        h.attrs["OmegaBaryon"] = 0.045
        h.attrs["HubbleParam"] = 0.7
        u = f.create_group("Units")
        u.attrs["UnitLength_in_cm"] = 3.085678e21
        u.attrs["UnitMass_in_g"] = 1.989e43
        u.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        u.attrs["UnitTime_in_s"] = 3.085678e16
        g = f.create_group("PartType1")
        g.create_dataset("ParticleIDs", data=np.arange(i*NPART, (i+1)*NPART, dtype=np.uint64), **kw)
        g.create_dataset("Coordinates",
                         data=(rng.random((NPART, 3)) * 100000).astype(np.float32), **kw3)
        g.create_dataset("Velocities",
                         data=(rng.standard_normal((NPART, 3)) * 100).astype(np.float32), **kw3)
        g.create_dataset("Masses", data=np.full(NPART, 1e-3, dtype=np.float32), **kw)
    print("wrote", i, flush=True)
