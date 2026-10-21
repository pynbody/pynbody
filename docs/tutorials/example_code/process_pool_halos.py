import pynbody


def analyse(f, h, halo_number):
    """Analyse one halo. This runs in one of the pool's processes, which has its own copy of f and h."""
    halo = h[halo_number]
    centre = pynbody.analysis.halo.shrink_sphere_center(halo)

    # The KD-tree speeds up spatial queries on the whole snapshot. It is built once, by the main process, the first
    # time any process asks for it
    f.build_tree()
    mass_nearby = f[pynbody.filt.Sphere('100 kpc', centre)]['mass'].sum()

    # To centre or rotate a halo, first make a copy of it; this process's f and h are read-only
    halo = halo.get_copy_on_access_simsnap()
    pynbody.analysis.center(halo)
    profile = pynbody.analysis.Profile(halo.dm, ndim=3, rmin='0.5 kpc', rmax='50 kpc', nbins=20)

    return mass_nearby, profile['density']


if __name__ == '__main__':
    f = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')
    f.physical_units()
    h = f.halos()

    # Analyse the first nine halos, four at a time
    with pynbody.ProcessPool(f, h, processes=4) as pool:
        results = pool.map(analyse, range(1, 10))

    for halo_number, (mass_nearby, density) in zip(range(1, 10), results):
        print(f"Halo {halo_number}: {mass_nearby:.2e} within 100 kpc; central density {density[0]:.2e}")
