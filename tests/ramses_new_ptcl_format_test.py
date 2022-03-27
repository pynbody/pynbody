import glob
import os
import warnings

import numpy as np

import pynbody

sink_filename = "testdata/ramses_new_format_partial_output_00001/sink_00001.csv"
sink_filename_moved = sink_filename+".temporarily_moved"

import pytest


def setup_module():
    global f
    f = pynbody.load("testdata/ramses_new_format_partial_output_00001")

def test_family_lengths():
    assert len(f.dm)==136006
    assert len(f.star)==12236
    assert len(f.gas)==196232
    assert len(f.bh)==2
    assert pynbody.family.bh in f.families()

def test_properties():
    np.testing.assert_almost_equal(f.properties['a'], 1.0)
    np.testing.assert_almost_equal(f.properties['h'], 0.01)
    np.testing.assert_almost_equal(f.properties['omegaM0'], 1.0)

def test_sink_variables():
    np.testing.assert_allclose(f.bh['pos'], [[2.5e-2,2.5e-2,2.5e-2],
                                             [3.5e-2,4.5e-2,5.5e-2]])
    assert str(f.bh['pos'].units)=="3.09e+21 cm"
    assert (f.bh['id']==np.array([1,2])).all()

def _unexpected_format_warning_was_issued(warnings):
    return any(["unexpected format" in str(w_i).lower() for w_i in warnings])

def _no_bh_family_present(f):
    return (len(f.bh)==0) and (pynbody.family.bh not in f.families())

def test_no_sink_file():
    try:
        os.rename(sink_filename, sink_filename_moved)
        with warnings.catch_warnings(record=True) as w:
            f_no_sink = pynbody.load("testdata/ramses_new_format_partial_output_00001")
        assert not _unexpected_format_warning_was_issued(w)
        assert _no_bh_family_present(f_no_sink)
    finally:
        os.rename(sink_filename_moved, sink_filename)

def test_garbled_sink_file():
    try:
        os.rename(sink_filename, sink_filename_moved)
        with open(sink_filename,"w") as tfile:
            tfile.write("1,2,3\r\n")

        with warnings.catch_warnings(record=True) as w:
            f_garbled_sink = pynbody.load("testdata/ramses_new_format_partial_output_00001")
        assert _unexpected_format_warning_was_issued(w)
        assert _no_bh_family_present(f_garbled_sink)

        with open(sink_filename,"w") as tfile:
            for i in range(4):
                tfile.write("1,2,3\r\n")

        f_garbled_sink = pynbody.load("testdata/ramses_new_format_partial_output_00001")
        # Would be nice to test the warning is also raised here, but need to figure out how to get python to
        # re-raise it despite the fact it was already triggered above.

        assert _no_bh_family_present(f_garbled_sink)
    finally:
        os.rename(sink_filename_moved, sink_filename)

def test_load_pos():
    loaded_vals = f.dm['pos'][::5001]
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

def test_load_iord():
    loaded_vals = f.star['iord'][::1001]
    np.testing.assert_equal(loaded_vals,
                            [1001684, 1006393, 1039350, 1099620, 1035230, 1086365, 1000185,
                             1000429, 1077059, 1011855, 1096842, 1013422, 1020021])

def _make_virtual_output_with_no_ptcls():
    if os.path.exists("testdata/ramses_new_format_partial_no_ptcls_output_00001"):
        return

    os.mkdir("testdata/ramses_new_format_partial_no_ptcls_output_00001")
    for i in glob.glob("testdata/ramses_new_format_partial_output_00001/*"):
        if "part_00001" not in i and '.csv' not in i:
            os.symlink("../../"+i, i.replace("ramses_new_format_partial_output","ramses_new_format_partial_no_ptcls_output"))

def test_load_no_ptcls():
    _make_virtual_output_with_no_ptcls()
    f = pynbody.load("testdata/ramses_new_format_partial_no_ptcls_output_00001")
    assert len(f.dm)==0
    assert len(f.star)==0
    assert len(f.gas) == 196232

def test_loaded_namelist():
    assert f._namelist['cosmo'] == False
    assert f._namelist['hydro'] == True
    assert f._namelist['z_ave'] == 0.1
    assert f._namelist.get("non_existent_key") == None
