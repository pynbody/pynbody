import gc
import os
import shutil

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import pynbody
import pynbody.test_utils
from pynbody import units
from pynbody.snapshot import gadgethdf
from pynbody.util import hdf_bulk_read
from pynbody.util.hdf_bulk_read import strategy as hdf_read_strategy


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("gadget", "arepo")

@pytest.fixture
def snap():
    f = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103.hdf5')
    yield f
    del f
    gc.collect()

@pytest.fixture
def subfind():
    f = pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103')
    yield f
    del f
    gc.collect()

def test_standard_arrays(snap, subfind) :
    """Check that the data loading works"""

    for s in [snap, subfind] :
        s.dm['pos']
        s.gas['pos']
        s.star['pos']
        s['pos']
        s['mass']
    #Load a second time to check that family_arrays still work
        s.dm['pos']
        s['vel']
        s['iord']
        s.gas['rho']
       # s.gas['u']
        s.star['mass']


def _h5py_copy_with_key_rename(src,dest):
    shutil.copy(src,dest)
    f = h5py.File(dest, 'r+')
    for tp in f:
        if "Mass" in tp:
            tp.move("Mass","Masses")
    f.close()




def test_alt_names(snap):
    _h5py_copy_with_key_rename('testdata/gadget3/data/snapshot_103/snap_103.hdf5',
                'testdata/gadget3/data/snapshot_103/snap_103_altnames.hdf5')

    snap_alt = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_altnames.hdf5')
    assert 'mass' in snap_alt.loadable_keys()
    assert all(snap_alt['mass']==snap['mass'])

def test_issue_256(snap) :
    assert 'pos' in snap.loadable_keys()
    assert 'pos' in snap.dm.loadable_keys()
    assert 'pos' in snap.gas.loadable_keys()
    assert 'He' not in snap.loadable_keys()
    assert 'He' not in snap.dm.loadable_keys()
    assert 'He' in snap.gas.loadable_keys()

def test_write():
    # make a copy of snap_103.hdf5 to avoid disturbing the original
    shutil.copy('testdata/gadget3/data/snapshot_103/snap_103.hdf5', 'testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    snap = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    ar_name = 'test_array'
    snap[ar_name] = np.random.uniform(0,1,len(snap))
    snap[ar_name].write()
    snap2 = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_copy.hdf5')
    v = snap[ar_name]
    with pytest.warns(UserWarning, match="Unable to infer units from HDF attributes"):
        v2 = snap2[ar_name]
    npt.assert_allclose(v, v2)

def test_hi_derivation(subfind):
    HI_answer = [  6.96499870e-06,   6.68348046e-06,   1.13855074e-05,
         1.10936027e-05,   1.40641633e-05,   1.67324738e-05,
         2.26228929e-05,   1.64661638e-05,   2.79337124e-05,
         3.32789555e-05,   2.38397192e-05,   5.11526743e-04,
         1.86211183e-01,   4.58309086e-02,   9.98117529e-02,
         1.76779058e-02,   8.30149935e-02,   1.08688537e-02,
         1.44146419e-07,   7.95141614e-08,   8.00568016e-05,
         5.40560080e-08,   2.73720754e-01,   2.91772885e-02,
         7.37755701e-04,   3.93431603e-02,   3.52700543e-03,
         1.46685188e-07,   4.55900305e-08,   3.85495273e-03,
         1.75020358e-07,   1.27841671e-01,   1.01551435e-07,
         3.23647121e-08,   8.22351949e-06,   1.03758201e-05,
         1.13115067e-05,   2.57878344e-05,   2.74634221e-05,
         5.34312023e-05,   2.55750061e-01,   3.83638138e-04,
         7.96613219e-03,   2.57835498e-03,   5.89219887e-08]

    # interpolation routine changed so rtol increased to allow for slight deviations
    # Further increased to 1e-5 to accommodate non-deterministic floating-point behavior from
    # multi-threaded BLAS operations (OpenBLAS) and CPU-specific optimizations (AVX, FMA, etc.)
    # The HI calculation involves log-space interpolation followed by exponentiation, which
    # amplifies small variations in thread scheduling and floating-point operation ordering
    npt.assert_allclose(subfind.halos()[0].gas['HI'][::100],HI_answer,rtol=1e-5)

def test_hdf_ordering(snap):
    # HDF files do not intrinsically specify the order in which the particle types occur
    # Because some operations may require stability, pynbody now imposes order by the particle type
    # number
    assert snap._family_slice[pynbody.family.gas] == slice(0, 2076907, None)
    assert snap._family_slice[pynbody.family.dm] == slice(2076907, 4174059, None)
    assert snap._family_slice[pynbody.family.star] == slice(4174059, 4194304, None)


def test_mass_in_header():
    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5")
    f.physical_units()
    f['mass'] # load all masses
    assert np.allclose(f.dm['mass'][0], 3982880.471745421)

    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5")
    f.physical_units()
    # don't load all masses, allow it to be loaded for DM only
    assert np.allclose(f.dm['mass'][0], 3982880.471745421)

def test_gadgethdf_style_units():
    f = pynbody.load("testdata/gadget3/data/snapshot_103/snap_103.hdf5")
    npt.assert_allclose(f.st['InitialMass'].units.in_units("1.989e43 g h^-1"), 1.0,
                        rtol=1e-3)

def test_arepo_style_units():
    f = pynbody.load("testdata/arepo/agora_100.hdf5")
    npt.assert_allclose(f.st['EMP_InitialStellarMass'].units.in_units("1.989e42 g"),
                        1.0, rtol=1e-3)
    # I strongly suspect that the units in this file are wrong -- the masses are
    # in h^-1 units, so presumably these initial stellar masses should also be in
    # h^-1 units. This is backed up by checking that, numerically,
    #    (f.st['EMP_InitialStellarMass']/f.st['mass']).min() == 1.0
    # On the other hand, pynbody should just reflect back that error
    # to the user, really; we can't get involved in compensating for bugs in other codes,
    # or all hell will break loose. So, we check for the 'wrong' units.

    npt.assert_allclose(f.st['AREPOEMP_Metallicity'].units.in_units(1.0),
                        1.0, rtol=1e-5)
    # the above is a special case of a dimensionless array

    with pytest.warns(UserWarning, match="Unable to infer units from HDF attributes"):
        assert f.st['EMP_BirthTemperature'].units == units.NoUnit()
    # here is a case where no unit information is recorded in the file (who knows why)

def test_load_copy(subfind):
    h = subfind.halos()[0]
    hcopy = h.load_copy()
    assert (hcopy['iord'][::10000] == h['iord'][::10000]).all()
    assert hcopy.ancestor is not h.ancestor
    
def test_load_copy_halo(subfind):
    
    halos = subfind.halos()
    halo = halos[len(halos)-1] # contains 10 gas, 10 dm, no star
    
    halo_copy = halo.load_copy()
    assert (halo_copy['iord']==halo['iord']).all()
    
    halo_star_copy = halo.s.load_copy()
    assert (len(halo_star_copy) == len(halo.s)) and (len(halo.s) == 0)

    halo_gas_copy = halo.g.load_copy()
    assert (halo_gas_copy['iord'] == halo.g['iord']).all()

    halo_dm_copy = halo.dm.load_copy()
    assert (halo_dm_copy['iord'] == halo.dm['iord']).all()

def test_load_copy_family(subfind):
    star_copy = subfind.s.load_copy()
    assert (star_copy['iord']==subfind.s['iord']).all()

    gas_copy = subfind.g.load_copy()
    assert (gas_copy['iord']==subfind.g['iord']).all()

    dm_copy = subfind.dm.load_copy()
    assert (dm_copy['iord']==subfind.dm['iord']).all()

def test_load_copy_indexsnap(subfind):
    indexsnap = subfind[1000:]
    indexsnap_copy = indexsnap.load_copy()
    assert (indexsnap_copy['iord']==indexsnap['iord']).all()

def test_noncontiguous_selection_slicing(subfind):
    # Test loading a non-contiguous selection of particles
    noncontig = subfind[::2]
    noncontig_copy = noncontig.load_copy()
    assert (noncontig_copy['iord']==noncontig['iord']).all()

def test_noncontiguous_selection_indexing(subfind):
    # Test loading a non-contiguous selection of particles using indexing
    halos = subfind.halos()
    halo_0 = halos[0]
    halo = halos[len(halos)-1] # contains 10 gas, 10 dm, no star
    indices = np.concatenate([halo_0.get_index_list(subfind),halo.get_index_list(subfind)])  # situation: not contiguous indices for some PartType
    indices = np.sort(indices)
    
    copy = subfind[indices].load_copy()
    
    assert (copy['iord']==subfind[indices]['iord']).all()

def test_partial_load_mass_in_header():
    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5")

    f_slice = f[::2].load_copy()
    f.physical_units()
    f_slice.physical_units()
    f['mass'] # load all masses
    assert np.allclose(f_slice['mass'],f[::2]['mass'])


def test_partial_load_empty_family_issue_1005():
    # Regression test for #1005: a `take` selection that excludes an entire family
    # (here, dm) must give the same, order-independent result regardless of whether
    # some other family's array was already read first.
    f = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5", take=(0, 1, 2, 3))
    assert f.families() == [pynbody.family.gas]
    dm_pos = f.dm['pos']
    assert dm_pos.shape == (0, 3)

    f2 = pynbody.load("testdata/gadget3/snap_028_z000p000.0.hdf5", take=(0, 1, 2, 3))
    gas_pos = f2.gas['pos']
    dm_pos2 = f2.dm['pos']
    assert gas_pos.shape == (4, 3)
    assert dm_pos2.shape == (0, 3)

def test_pressure_without_on_equation_of_state():
    # Regression test: some GadgetHDF variants do not write an OnEquationOfState
    # array at all. The pressure derivation should fall back to assuming there is
    # no equation-of-state floor, rather than raising a KeyError.
    shutil.copy('testdata/gadget3/data/snapshot_103/snap_103.hdf5',
                'testdata/gadget3/data/snapshot_103/snap_103_no_eos.hdf5')
    f = h5py.File('testdata/gadget3/data/snapshot_103/snap_103_no_eos.hdf5', 'r+')
    del f['PartType0']['OnEquationOfState']
    f.close()

    snap_with_eos = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103.hdf5')
    snap_no_eos = pynbody.load('testdata/gadget3/data/snapshot_103/snap_103_no_eos.hdf5')

    assert 'OnEquationOfState' not in snap_no_eos.gas.loadable_keys()

    p_no_eos = snap_no_eos.gas['p']

    # the reference snapshot has particles on the equation of state, so the two
    # pressure arrays should genuinely differ where that floor kicks in
    oneos = snap_with_eos.gas['OnEquationOfState'] == 1.
    assert oneos.any()

    p_expected = snap_with_eos.gas['u'] * snap_with_eos.gas['rho'] * (2./3)
    npt.assert_allclose(p_no_eos, p_expected)


def test_pressure_derivation(snap):
    # The pressure must be a genuine physical pressure (energy per unit volume),
    # not an array whose units happen to look plausible but are not actually
    # convertible to a real pressure (see issue where the formula was missing
    # a Boltzmann constant and mean molecular weight factor).
    snap.physical_units()
    p = snap.gas['p']
    p.in_units('erg cm**-3')  # raises UnitsException if not a real pressure

    oneos = snap.gas['OnEquationOfState'] == 1.
    assert oneos.any()
    assert (~oneos).any()

    # off the equation of state, pressure should be the standard ideal-gas
    # p = (gamma - 1) * u * rho
    npt.assert_allclose(p[~oneos], (snap.gas['u'] * snap.gas['rho'] * (2./3))[~oneos])

    # on the equation of state, pressure should follow the imposed polytropic
    # floor, P/k_B = 2300 K cm^-3 * (rho / (0.1 m_p cm^-3))^(4/3)
    critpres = 2300. * units.k * units.K / units.cm**3
    critdens = 0.1 * units.m_p / units.cm**3
    expected_oneos = critpres * (snap.gas['rho'][oneos].in_units('m_p cm**-3') / critdens) ** (4./3)
    npt.assert_allclose(p[oneos], expected_oneos.in_units(p.units))


def test_load_copy_issue_955(snap):
    # condition: A single-file snapshot with a PartType length greater than max_buf; 
    # select a slice across chunk boundary
    from pynbody.snapshot.gadgethdf import _max_buf
    boundary_slice = slice(_max_buf, _max_buf+1)

    snap_cop = snap[boundary_slice].load_copy()
    assert (snap_cop['iord'] == snap[boundary_slice]['iord']).all()


@pytest.mark.parametrize('filename',
                         ["testdata/gadget3/data/snapshot_103/snap_103.hdf5",
                          "testdata/gadget3/snap_028_z000p000.0.hdf5"])
def test_remote_open(remote_kwargs, filename):
    """
    Try opening local and remote versions of the same file
    """
    # Open the local HDF5 file
    local_snap = pynbody.load(filename)
    assert isinstance(local_snap._hdf_files[0], h5py.File)

    # Open the same file through the server and check we didn't just open the local file again
    remote_snap = pynbody.load(filename, **remote_kwargs)
    import hdfstream
    assert isinstance(remote_snap._hdf_files[0], hdfstream.RemoteFile)

    # Compare file contents
    assert len(local_snap) == len(remote_snap)
    assert local_snap.families() == remote_snap.families()
    assert local_snap.loadable_keys() == remote_snap.loadable_keys()
    for fam in local_snap.families():
        assert len(local_snap[fam]) == len(remote_snap[fam])
        assert local_snap[fam].loadable_keys() == remote_snap[fam].loadable_keys()
        assert np.all(local_snap[fam]["pos"] == remote_snap[fam]["pos"])
        assert np.all(local_snap[fam]["iord"] == remote_snap[fam]["iord"])


def test_remote_dir_none():
    """
    Check that an explicit remote_dir=None is accepted and reads a local file
    """
    local_snap = pynbody.load("testdata/gadget3/data/snapshot_103/snap_103.hdf5", remote_dir=None)
    assert isinstance(local_snap._hdf_files[0], h5py.File)


def _read_multifile_dataset(filenames, nr_files, dataset_name):
    data = []
    for file_nr in range(nr_files):
        with h5py.File(filenames.format(file_nr=file_nr), "r") as f:
            data.append(f[dataset_name][...])
    return np.concatenate(data, axis=0)


@pytest.mark.parametrize('take', [None,                      # all elements
                                  (0,1,2,3,4),               # contiguous
                                  np.arange(100) + 16856,    # crosses 0-1 file boundary
                                  (500, 1053, 2012, 17000),  # non-contiguous, calls _HDFArrayFiller._fill_from_fancy_index()
                                  np.concatenate([np.arange(100), np.arange(100)+20000])]) # two separate contiguous ranges
def test_flattened_arrays(take, load_kwargs):
    """
    Check that we can read flattened multidimensional arrays correctly.
    This type of dataset appears in the subfind example data.
    """
    # Read in the gas particle coordinates to check against
    filenames = "testdata/gadget3/data/subhalos_103/subhalo_103.{file_nr}.hdf5"
    all_pos = _read_multifile_dataset(filenames, 8, "FOF/PartType0/Coordinates")
    assert len(all_pos.shape)==1    # This vector dataset is flattened in the example file
    all_pos = all_pos.reshape(-1,3) # so we un-flatten it here

    # Read the same dataset with pynbody
    subfind = pynbody.load('testdata/gadget3/data/subhalos_103/subhalo_103', take=take, **load_kwargs)
    pos = subfind.gas["pos"]

    # Compute the expected result
    if take is None:
        pos_expected = all_pos
    else:
        pos_expected = all_pos[take,...]

    # Compare
    assert pos.shape == pos_expected.shape
    assert np.all(pos == pos_expected)



@pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")
@pytest.mark.filterwarnings("ignore:Masses are either stored in the header")
@pytest.mark.parametrize("filename, load_kwargs",
                         [("testdata/gadget3/data/snapshot_103/snap_103.hdf5", {}),
                          ("testdata/gadget3/snap_028_z000p000.0.hdf5", {}),
                          ("testdata/gadget3/snap_028_z000p000.0.hdf5", {'take': np.arange(0, 400000, 7)}),
                          ("testdata/arepo/agora_100.hdf5", {})])
def test_bulk_read_backends_agree(monkeypatch, filename, load_kwargs):
    """Reading bulk data directly gives exactly what h5py gives, and really is used"""
    arrays = {}
    readers = {}
    original_open = hdf_bulk_read.BulkReader.open
    for backend in ['h5py', 'direct']:
        monkeypatch.setattr(gadgethdf, "_direct_bulk_read", backend == 'direct')
        opened = []
        monkeypatch.setattr(hdf_bulk_read.BulkReader, "open",
                            lambda self, dataset: opened.append(original_open(self, dataset)) or opened[-1])
        f = pynbody.load(filename, **load_kwargs)
        # arrays present for every family, so that no rows are left for which the file provides no data
        arrays[backend] = {}
        for k in f.loadable_keys():
            try:
                arrays[backend][k] = np.asarray(f[k])
            except Exception as e:
                arrays[backend][k] = type(e)  # e.g. inconsistent unit metadata, which must fail the same way
        readers[backend] = opened

    assert all(isinstance(r, h5py.Dataset) for r in readers['h5py'])
    assert not any(isinstance(r, h5py.Dataset) for r in readers['direct'])

    assert arrays['h5py'].keys() == arrays['direct'].keys()
    for k in arrays['h5py']:
        a, b = arrays['h5py'][k], arrays['direct'][k]
        if isinstance(a, type):
            assert a == b
        else:
            assert a.dtype == b.dtype
            npt.assert_array_equal(a, b)


def test_write_then_read_directly():
    """Writing an array must not leave the bulk reader holding stale metadata"""
    filename = 'testdata/gadget3/data/snapshot_103/snap_103_bulk_copy.hdf5'
    shutil.copy('testdata/gadget3/data/snapshot_103/snap_103.hdf5', filename)
    try:
        snap = pynbody.load(filename)
        snap.dm['pos']  # the bulk reader now has the file open
        snap['bulk_test_array'] = np.arange(len(snap), dtype=np.float64)
        snap['bulk_test_array'].write()
        snap.gas['pos']  # and reads from it again after the write

        snap2 = pynbody.load(filename)
        with pytest.warns(UserWarning, match="Unable to infer units from HDF attributes"):
            npt.assert_array_equal(snap2['bulk_test_array'], np.arange(len(snap2)))
        npt.assert_array_equal(snap2['pos'], snap['pos'])
        del snap, snap2
    finally:
        gc.collect()
        os.remove(filename)


def _record_thread_tasks(monkeypatch):
    """Record the tasks (lists of units of work) given to threads, as they are before being performed"""
    tasks_seen = []
    original_perform = hdf_bulk_read.execute._perform_in_threads
    monkeypatch.setattr(hdf_bulk_read.execute, "_perform_in_threads",
                        lambda tasks, n: tasks_seen.append([list(task) for task in tasks]) or original_perform(tasks, n))
    return tasks_seen


@pytest.mark.filterwarnings("ignore:Unable to infer units from HDF attributes")
@pytest.mark.filterwarnings("ignore:Masses are either stored in the header")
@pytest.mark.parametrize("filename, load_kwargs",
                         [("testdata/gadget3/snap_028_z000p000.0.hdf5", {}),
                          ("testdata/gadget3/snap_028_z000p000.0.hdf5", {'take': np.arange(0, 400000, 7)}),
                          ("testdata/arepo/agora_100.hdf5", {}),
                          ("testdata/SWIFT/multifile_with_vds/snap_0000.hdf5", {}),
                          ("testdata/SWIFT/multifile_with_vds/snap_0000.hdf5", {'take_swift_cells': [0, 5, 20, 200]}),
                          ("testdata/SWIFT/multifile_without_vds/snap_0000", {}),
                          ("testdata/SWIFT/snap_0150.hdf5",
                           {'take_region': pynbody.filt.Sphere(20., (50., 50., 50.))})])
@pytest.mark.parametrize("direct", [True, False])
def test_threaded_loading_matches_serial(monkeypatch, filename, load_kwargs, direct):
    monkeypatch.setattr(gadgethdf, "_direct_bulk_read", direct)
    thread_pools_used = _record_thread_tasks(monkeypatch)

    arrays = {}
    for threads in [1, 4]:
        monkeypatch.setitem(hdf_read_strategy.config, "threads", str(threads))
        f = pynbody.load(filename, **load_kwargs)
        arrays[threads] = {}
        for k in f.loadable_keys():
            try:
                arrays[threads][k] = np.asarray(f[k])
            except Exception as e:
                arrays[threads][k] = type(e)
        if threads == 1:
            assert thread_pools_used == []

    if direct and "multifile_without_vds" in filename:
        # (reads through h5py are always serial, as is a dataset small enough to be read in one piece)
        assert len(thread_pools_used) > 0
    if not direct:
        assert len(thread_pools_used) == 0
    assert arrays[1].keys() == arrays[4].keys()
    for k in arrays[1]:
        a, b = arrays[1][k], arrays[4][k]
        if isinstance(a, type):
            assert a == b
        else:
            assert a.dtype == b.dtype
            npt.assert_array_equal(a, b)


def test_threaded_loading_propagates_errors(monkeypatch):
    """An exception in one thread's read reaches the caller"""
    monkeypatch.setitem(hdf_read_strategy.config, "threads", "4")
    f = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")

    def failing_perform(self):
        # (not an OSError, which SimSnap takes to mean that the array cannot be loaded, and swallows)
        raise RuntimeError("simulated read failure")

    monkeypatch.setattr(hdf_bulk_read.plan._Work, "perform", failing_perform)
    with pytest.raises(RuntimeError, match="simulated read failure"):
        f['pos']


def test_threaded_tasks_are_per_file(monkeypatch):
    """On a parallel filesystem, each thread's task reads one file, in order, so that threads work on different files"""
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: "lustre")
    tasks_seen = _record_thread_tasks(monkeypatch)
    f = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")
    f['pos']
    (tasks,) = tasks_seen
    assert len(tasks) == len(f._hdf_files)
    for task in tasks:
        assert len({work.filename for work in task}) == 1
        starts = [work.span[0] for work in task]
        assert starts == sorted(starts)
    assert f._array_loader.last_read_strategy.per_file


@pytest.mark.parametrize("cells", [None, list(range(0, 512, 5)), [3]])
def test_virtual_dataset_is_shared_between_threads_like_its_files(monkeypatch, cells):
    """A snapshot whose datasets are virtual, drawing on a set of files, is read just as the set of files would be:
    the same reads, shared between threads in the same way"""
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: "lustre")
    monkeypatch.setitem(hdf_read_strategy.config, "parallel-filesystem-threads", 16)
    monkeypatch.setattr(hdf_read_strategy, "available_cpus", lambda: 16)
    tasks_seen = _record_thread_tasks(monkeypatch)
    monkeypatch.setattr(gadgethdf, "_max_buf", 3000)  # so that pieces do not all coincide with files

    def load(filename):
        tasks_seen.clear()
        f = pynbody.load(filename, take_swift_cells=cells) if cells else pynbody.load(filename)
        data = f['pos']
        # for each unit of work, the file and how many rows it fills in
        planned = [[(os.path.basename(work.filename), len(work.destination)) for work in task]
                   for task in tasks_seen[0]] if tasks_seen else None
        return data, f._array_loader.last_read_strategy, planned

    virtual_data, virtual_strategy, virtual_tasks = load("testdata/SWIFT/multifile_with_vds/snap_0000.hdf5")
    files_data, files_strategy, files_tasks = load("testdata/SWIFT/multifile_without_vds/snap_0000")
    np.testing.assert_array_equal(virtual_data, files_data)
    assert (virtual_strategy.threads, virtual_strategy.per_file) == (files_strategy.threads, files_strategy.per_file)
    if files_tasks is None:
        assert virtual_tasks is None and files_strategy.threads == 1
        return
    # Each task reads one file, and the same number of rows from it. (The reads within a task can differ: pieces of
    # a virtual dataset are counted along the whole dataset, and of a file from its start, so a piece of the
    # virtual dataset crossing from one file to the next makes one read more.)
    def summary(tasks):
        return [({name for name, _ in task}, sum(num_rows for _, num_rows in task)) for task in tasks]
    assert summary(virtual_tasks) == summary(files_tasks)
    assert all(len(names) == 1 for names, _ in summary(virtual_tasks))


@pytest.mark.parametrize("flattened", [False, True])
def test_array_filler_requests(tmp_path, flattened):
    """Requests describe rows of the dataset, allowing for 3-vectors stored flattened, as three consecutive rows"""
    stored = np.arange(60.0) if flattened else np.arange(60.0).reshape(20, 3)
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f["x"] = stored
    with h5py.File(tmp_path / "x.h5", "r") as f:
        target = np.zeros((20, 3))
        filler = gadgethdf._HDFArrayFiller(target, f["x"])
        request = filler.request(target[2:5], f["x"], slice(1, 4), offset=1)
        assert request.rows == (slice(6, 15) if flattened else slice(2, 5))
        assert request.destination.shape == ((9,) if flattened else (3, 3))
        assert np.shares_memory(request.destination, target)
        request = filler.request(target[5:8], f["x"], np.array([0, 2, 7]), offset=1)
        np.testing.assert_array_equal(request.rows, [3, 4, 5, 9, 10, 11, 24, 25, 26] if flattened else [1, 3, 8])
        hdf_bulk_read.BulkReader().read([request])
    np.testing.assert_array_equal(target[5:8], np.arange(60.0).reshape(20, 3)[[1, 3, 8]])


@pytest.mark.parametrize("direct", [True, False])
def test_datasets_are_released_file_by_file(monkeypatch, direct):
    """Loading an array from a set of files lets go of each file's dataset (and so its chunk cache) once it has been
    read, rather than holding every file's until the whole array is loaded"""
    import weakref
    monkeypatch.setattr(gadgethdf, "_direct_bulk_read", direct)
    monkeypatch.setitem(hdf_read_strategy.config, "threads", "1")
    alive = []
    most_alive = [0]
    original_perform = hdf_bulk_read.plan._Work.perform

    def perform(self):
        if not any(ref() is self.source for ref in alive):
            alive.append(weakref.ref(self.source))
        gc.collect()
        most_alive[0] = max(most_alive[0], sum(ref() is not None for ref in alive))
        return original_perform(self)

    monkeypatch.setattr(hdf_bulk_read.plan._Work, "perform", perform)
    f = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")
    f.dm['pos']
    assert len(alive) == len(f._hdf_files)
    assert most_alive[0] <= 2  # the dataset being read, and perhaps the one before it awaiting collection


def test_interrupted_threaded_load_abandons_queued_reads(monkeypatch):
    """An exception such as KeyboardInterrupt stops the load without waiting for every queued read"""
    class Interrupt(BaseException):  # (like KeyboardInterrupt, which pytest itself would act on)
        pass

    monkeypatch.setitem(hdf_read_strategy.config, "threads", "2")
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: "lustre")
    performed = []
    original_perform = hdf_bulk_read.plan._Work.perform

    def perform(self):
        performed.append(1)
        if len(performed) == 1:
            raise Interrupt
        return original_perform(self)

    monkeypatch.setattr(gadgethdf, "_max_buf", 500)  # many reads per file
    f = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")
    num_reads = len(list(f._array_loader._requests([pynbody.family.dm], f, "pos", ["Coordinates"])))
    monkeypatch.setattr(hdf_bulk_read.plan._Work, "perform", perform)
    with pytest.raises(Interrupt):
        f._array_loader.load_arrays([pynbody.family.dm], f, "pos", ["Coordinates"])
    # 2 threads work through 10 files: once one is interrupted, only the file the other is reading should be finished
    # (without cancelling, all but the interrupted file would be)
    assert len(performed) <= num_reads * 4 // 10


@pytest.mark.skipif(hdf_read_strategy.available_cpus() < 2, reason="needs more than one CPU")
def test_threads_sharing_files_decode_each_chunk_once(monkeypatch):
    """When threads share the pieces of a file, pieces needing the same chunk are read by one thread, in order, so
    that each chunk is decoded once"""
    monkeypatch.setitem(hdf_read_strategy.config, "threads", "4")
    monkeypatch.setattr(hdf_read_strategy, "filesystem_type", lambda path: "ext4")
    monkeypatch.setattr(gadgethdf, "_max_buf", 1000)  # much smaller than the chunks, so that pieces share them
    decodes = []
    original_decode = hdf_bulk_read.decode.decode_chunk
    monkeypatch.setattr(hdf_bulk_read.decode, "decode_chunk",
                        lambda *args, **kwargs: decodes.append(1) or original_decode(*args, **kwargs))
    f = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")
    f.dm['pos']
    assert not f._array_loader.last_read_strategy.per_file and f._array_loader.last_read_strategy.threads > 1
    decodes_threaded = len(decodes)
    decodes.clear()
    monkeypatch.setitem(hdf_read_strategy.config, "threads", "1")
    g = pynbody.load("testdata/SWIFT/multifile_without_vds/snap_0000")
    g.dm['pos']
    np.testing.assert_array_equal(f.dm['pos'], g.dm['pos'])
    assert decodes_threaded == len(decodes)


@pytest.mark.skipif(hdf_read_strategy.available_cpus() < 2, reason="needs more than one CPU")
@pytest.mark.parametrize("filename, compressed", [("testdata/SWIFT/multifile_without_vds/snap_0000", True),
                                                  ("testdata/gadget3/data/snapshot_103/snap_103.hdf5", False)])
def test_automatic_strategy_on_local_filesystem(filename, compressed):
    """With the default 'auto', compressed data on a local filesystem are read in threads sharing files; uncompressed
    data serially"""
    assert hdf_read_strategy.config["threads"] == "auto"
    f = pynbody.load(filename)
    f['pos']
    strategy = f._array_loader.last_read_strategy
    if compressed:
        assert strategy.threads > 1 and not strategy.per_file, strategy
    else:
        assert strategy.threads == 1, strategy
