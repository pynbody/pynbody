import glob
import os
import pathlib
import shutil

import numpy as np
import pytest

import pynbody
import pynbody.test_utils


@pytest.fixture(scope='module', autouse=True)
def get_data():
    pynbody.test_utils.ensure_test_data_available("ramses")


sink_filename = "testdata/ramses/ramses_new_format_partial_output_00001/sink_00001.csv"
sink_filename_moved = sink_filename+".temporarily_moved"

@pytest.fixture
def snap():
    return pynbody.load("testdata/ramses/ramses_new_format_partial_output_00001")

def test_family_lengths(snap):
    assert len(snap.dm) == 136006
    assert len(snap.star) == 12236
    assert len(snap.gas) == 196232
    assert len(snap.bh) == 2
    assert pynbody.family.bh in snap.families()

def test_properties(snap):
    np.testing.assert_almost_equal(snap.properties['a'], 1.0)
    np.testing.assert_almost_equal(snap.properties['h'], 0.01)
    np.testing.assert_almost_equal(snap.properties['omegaM0'], 1.0)

def test_sink_variables(snap):
    np.testing.assert_allclose(snap.bh['pos'], [[2.5e-2, 2.5e-2, 2.5e-2],
                                                [3.5e-2,4.5e-2,5.5e-2]])
    assert str(snap.bh['pos'].units) == "3.09e+21 cm"
    assert (snap.bh['id'] == np.array([1, 2])).all()


def _no_bh_family_present(f):
    return (len(f.bh)==0) and (pynbody.family.bh not in f.families())

@pytest.fixture
def backup_sinkfile():
    os.rename(sink_filename, sink_filename_moved)
    yield sink_filename
    try:
        os.remove(sink_filename)
    except OSError:
        pass
    os.rename(sink_filename_moved, sink_filename)


def test_no_sink_file(backup_sinkfile):
    # No warning should be raised
    f_no_sink = pynbody.load("testdata/ramses/ramses_new_format_partial_output_00001")
    assert _no_bh_family_present(f_no_sink)

def test_garbled_sink_file(backup_sinkfile):
    with open(sink_filename, "w") as tfile:
        tfile.write("1,2,3\r\n")

    warn_str = r"Unexpected format in file .*\.csv -- sink data has not been loaded"
    with pytest.warns(UserWarning, match=warn_str):
        f_garbled_sink = pynbody.load("testdata/ramses/ramses_new_format_partial_output_00001")
    assert _no_bh_family_present(f_garbled_sink)

    with open(sink_filename,"w") as tfile:
        for i in range(4):
            tfile.write("1,2,3\r\n")

    with pytest.warns(UserWarning, match=warn_str):
        f_garbled_sink = pynbody.load("testdata/ramses/ramses_new_format_partial_output_00001")

    assert _no_bh_family_present(f_garbled_sink)

def test_load_pos(snap):
    loaded_vals = snap.dm['pos'][::5001]
    test_vals = [[ 0.01105415,  0.02300947,  0.00724412],
       [ 0.02122232,  0.02265619,  0.00677652],
       [ 0.02018343,  0.02035065,  0.0095223 ],
       [ 0.02269471,  0.02136712,  0.00620482],
       [ 0.02287763,  0.01224741,  0.01737309],
       [ 0.01815873,  0.01128699,  0.02247262],
       [ 0.02065698,  0.00701042,  0.02248414],
       [ 0.01930041,  0.01295866,  0.01411238],
       [ 0.02427911,  0.01443727,  0.02082101],
       [ 0.02015258,  0.01861216,  0.01329177],
       [ 0.02016469,  0.01742381,  0.02150717],
       [ 0.02356023,  0.02185429,  0.01414604],
       [ 0.01941696,  0.02358911,  0.01747907],
       [ 0.02231799,  0.02193878,  0.02197457],
       [ 0.02463229,  0.02258142,  0.02418932],
       [ 0.0129505 ,  0.01908118,  0.0244578 ],
       [ 0.0170727 ,  0.01979101,  0.02121195],
       [ 0.01921728,  0.02407719,  0.02357735],
       [ 0.01402912,  0.02266267,  0.01451735],
       [ 0.01251412,  0.01753267,  0.01765179],
       [ 0.01846307,  0.01644008,  0.02239697],
       [ 0.01017412,  0.01582988,  0.02493915],
       [ 0.00980422,  0.02452824,  0.02011278],
       [ 0.01242898,  0.02346281,  0.01821639],
       [ 0.01668166,  0.01431568,  0.01471138],
       [ 0.0067898 ,  0.01912026,  0.02174331],
       [ 0.02104224,  0.01082964,  0.02712232],
       [ 0.02348118,  0.00900178,  0.028894  ]]

    np.testing.assert_allclose(loaded_vals,test_vals, atol=1e-7)

def test_load_iord(snap):
    loaded_vals = snap.star['iord'][::1001]
    np.testing.assert_equal(loaded_vals,
                            [1001684, 1006393, 1039350, 1099620, 1035230, 1086365, 1000185,
                             1000429, 1077059, 1011855, 1096842, 1013422, 1020021])

def _make_virtual_output_with_no_ptcls():
    if os.path.exists("testdata/ramses/ramses_new_format_partial_no_ptcls_output_00001"):
        return

    test_dir = pathlib.Path("testdata/ramses/ramses_new_format_partial_no_ptcls_output_00001")
    source_dir = pathlib.Path("testdata/ramses/ramses_new_format_partial_output_00001")
    if not test_dir.exists():
        test_dir.mkdir(parents=True)

    for i in source_dir.iterdir():
        if ".csv" in i.name or "part_00001" in i.name:
            # skip the sink file
            continue
        else:
            # copy files
            shutil.copy(i, test_dir / i.name)

def test_load_no_ptcls():
    _make_virtual_output_with_no_ptcls()
    f = pynbody.load("testdata/ramses/ramses_new_format_partial_no_ptcls_output_00001")
    assert len(f.dm)==0
    assert len(f.star)==0
    assert len(f.gas) == 196232

def test_loaded_namelist(snap):
    assert snap._namelist['cosmo'] is False
    assert snap._namelist['hydro'] is True
    assert snap._namelist['z_ave'] == 0.1
    assert snap._namelist.get("non_existent_key") is None

@pytest.fixture(params=[True, False], ids=['mass-column-present', 'no-mass-column-present'])
def snap_for_issue_771(request):
    """Aids test for issue #771. In that example, an output has a 'mass' column in the sink csv file. In our test data
    we instead have an 'msink' column. So we create a virtual version where we rename the column to 'mass'"""
    src_dir = pathlib.Path('testdata/ramses/ramses_new_format_partial_output_00001')

    if request.param:
        # Make the virtual version of the output with 'mass' in place of 'msink' column
        tgt_dir = pathlib.Path('testdata/ramses/ramses_new_format_partial_with_sink_mass_output_00001')

        if not tgt_dir.exists():
            # Create target directory if it doesn't exist
            tgt_dir.mkdir()

            # Iterate over all files in the source directory
            for src_file in src_dir.iterdir():
                tgt_file : pathlib.Path = tgt_dir / src_file.name

                # If the file is 'sink_00001.csv', copy and modify its contents
                if src_file.name == 'sink_00001.csv':
                    with src_file.open() as f_in, tgt_file.open('w') as f_out:
                        for line in f_in:
                            f_out.write(line.replace('msink', 'mass'))
                else:
                    shutil.copy(src_file, tgt_file)

        yield pynbody.load(tgt_dir)

        # Clean up
        #for tgt_file in tgt_dir.iterdir():
        #    tgt_file.unlink()
        #tgt_dir.rmdir()
    else:
        yield pynbody.load(src_dir)

@pytest.mark.parametrize("in_physical_units", [True, False],
                         ids=['physical', 'code'])
def test_sink_file_mass_all_particles(snap_for_issue_771, in_physical_units):
    # it should be OK if there isn't a mass column in the sinks
    #f2 = pynbody.load("testdata/ramses/ramses_new_format_partial_output_00001")
    #f2['mass']

    if in_physical_units:
        snap_for_issue_771.physical_units()

    snap_for_issue_771['mass']

    reference_array = snap_for_issue_771['mass']
    reference_array_gas = snap_for_issue_771.gas['mass']

    # now let's try reloading that and accessing things in another order
    f = pynbody.load(snap_for_issue_771.filename)
    f.gas['mass']
    if in_physical_units:
        f.physical_units()

    # this should be fine:
    np.testing.assert_allclose(reference_array_gas, f.gas['mass'])

    # now the gas mass derived array is already present, the code path via which the final mass array
    # is assembled is different (no need to re-derive the gas mass). But this can seem to result
    # in wrong results with physical_units.

    test_array = f['mass'] # completes by loading the particle masses from disk. Should leave gas mass untouched.

    np.testing.assert_allclose(reference_array, test_array)
