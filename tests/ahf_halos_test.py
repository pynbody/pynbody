import glob
import os.path
import shutil
import stat
import subprocess

import pynbody


def test_load_ahf_catalogue():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert len(h)==1411

def test_load_ahf_catalogue_non_gzipped():
    subprocess.call(["gunzip","testdata/g15784.lr.01024.z0.000.AHF_halos.gz"])
    try:
        f = pynbody.load("testdata/g15784.lr.01024")
        h = pynbody.halo.AHFCatalogue(f)
        assert len(h)==1411
    finally:
        subprocess.call(["gzip","testdata/g15784.lr.01024.z0.000.AHF_halos"])

def test_ahf_properties():
    f = pynbody.load("testdata/g15784.lr.01024")
    h = pynbody.halo.AHFCatalogue(f)
    assert h[1].properties['children']==[]
    assert h[1].properties['fstart']==23


def _setup_unwritable_ahf_situation():
    if os.path.exists("testdata/test_unwritable"):
        os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR | stat.S_IWUSR)
        shutil.rmtree("testdata/test_unwritable")
    os.mkdir("testdata/test_unwritable/")
    for fname in glob.glob("testdata/g15784*"):
        if "AHF_fpos" not in fname:
            os.symlink("../"+fname[9:], "testdata/test_unwritable/"+fname[9:])
    os.chmod("testdata/test_unwritable", stat.S_IRUSR | stat.S_IXUSR)


def test_ahf_unwritable():
    _setup_unwritable_ahf_situation()
    f = pynbody.load("testdata/test_unwritable/g15784.lr.01024")
    h = f.halos()
    assert len(h)==1411


def test_ramses_ahf_family_mapping_with_new_format():
    # Test Issue 691 where family mapping of AHF catalogues with Ramses new particle formats would go wrong
    f = pynbody.load("testdata/output_00110")
    halos = pynbody.halo.AHFCatalogue(f)

    assert len(halos) == 2719 # 2720 lines in AHF halos file
    print(halos._all_parts)

    # Load two halos and check that stars, DM and gas are correctly mapped by pynbody
    # Halo 1 is the main halo and has all three families, while other are random pickes
    halo_numbers = [1, 10, 128, 256]
    for halo_number in halo_numbers:
        halo = halos[halo_number]

    # Ok so problem comes from get_halo and likely load_ahf_particle_block
    with pynbody.util.open_("testdata/output_00110/output_00110_fullbox.tipsy.z0.031.AHF_particles") as file:
        fpos = halo.properties['fstart']
        print(fpos)
        file.seek(fpos, 0)
        ids = (halos._load_ahf_particle_block(file, halo.properties['npart']))
        assert(halos._use_iord == False)
        assert(isinstance(halos.base, pynbody.snapshot.ramses.RamsesSnap))

        from collections import Counter
        print(ids)
    #     import numpy as np
    #     ndm = f._get_family_slice('dm').stop
    #
    #     masked_dm = ids[np.where(ids < ndm)]
    #     print(masked_dm)
    #     print(len(masked_dm))
    #     masked_star = ids[np.where((ids >= ndm) & (ids < f._get_family_slice('st').stop))]
    #     print(masked_star)
        # Find the correct offset for gas tracers

        # Reproduce the initialisation of the IndexedSubSnap
        # findex = f._family_index()[ids]
        # print(findex)
        # print(Counter(findex))
        # import numpy as np
        # for i, fam in enumerate(f.ancestor.families()):
        #     ids = np.where(findex == i)[0]
        #     print(fam)
        #     print(ids)

        # print(f[ids].st)
        # print(ids.max())
        # print(f._get_family_slice('st').start)
        # print(f._get_family_slice('st').stop)
        # print(f._get_family_slice('dm').start)
        # print(f._get_family_slice('dm').stop)
        # print(f._get_family_slice('gas').start)
        # print(f._get_family_slice('gas').stop)
        #
        # famslice = halos.base._get_family_slice('st')
        # target = halos.base[famslice]
        # print(target)

    # print(halos.get_group_array(halo))
    # print(halos.make_grp())
    # print(halo.d['grp'])
    # print(halo.st['grp'])
        from collections import Counter
        print(Counter(halo.d['mass'].in_units("Msol")))

        # There should not be any extra families in the halo particles
        assert(all(fam in [pynbody.family.dm, pynbody.family.star, pynbody.family.gas] for fam in halo.families()))

        # AHF assigned the same number of particles to the halo
        # than its header, family by family
        assert(halo.properties['npart'] == len(halo))
        assert(halo.properties['n_star'] == len(halo.st))
        assert(halo.properties['n_gas'] == len(halo.g))
        ndm = halo.properties['npart'] - halo.properties['n_star'] - halo.properties['n_gas']
        assert(ndm == len(halo.d))

        # Derive some masses to check that we are identifying the right particles
        # dm_mass = halo.properties['Mhalo'] - halo.properties['M_star'] - halo.properties['M_gas']
        star_mass = halo.properties['M_star']
        gas_mass = halo.properties['M_gas']
        import numpy.testing as npt
        rtol = 1e-2 # We are not precise to per cent level with unit conversion through the different steps
        # npt.assert_allclose(dm_mass, halo.d['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(star_mass, halo.st['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(gas_mass, halo.g['mass'].sum().in_units("Msol"), rtol=rtol)
        npt.assert_allclose(halo.properties['Mhalo'], halo['mass'].sum().in_units("Msol"), rtol=rtol)

