import pynbody


def analyse(f, h, halo_number):
    """Analyse one halo. This runs in one of the pool's processes, which has its own copy of f and h."""
    halo = h[halo_number]
    centre = pynbody.analysis.halo.shrink_sphere_center(halo)

    # Spatial queries can use the whole snapshot, and use the tree that was built beforehand
    mass_nearby = f[pynbody.filt.Sphere('100 kpc', centre)]['mass'].sum()

    # To centre or rotate a halo, first make a copy of it; this process's f and h are read-only
    halo = halo.get_copy_on_access_simsnap()
    pynbody.analysis.center(halo)
    profile = pynbody.analysis.Profile(halo.dm, ndim=3, rmin='0.5 kpc', rmax='50 kpc', nbins=20)

    return mass_nearby, profile['density']


if __name__ == '__main__':
    # 1. Load the snapshot, and ask for its arrays to be put in shared memory
    f = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')
    f.enable_shared_arrays()

    # 2. Prepare everything the analysis needs
    f.physical_units()
    f['pos']
    f['vel']
    f['mass']
    h = f.halos()
    f.build_tree()

    # 3. Analyse the first nine halos, four at a time
    with pynbody.ProcessPool(f, h, processes=4) as pool:
        results = pool.map(analyse, range(1, 10))

    for halo_number, (mass_nearby, density) in zip(range(1, 10), results):
        print(f"Halo {halo_number}: {mass_nearby:.2e} within 100 kpc; central density {density[0]:.2e}")
