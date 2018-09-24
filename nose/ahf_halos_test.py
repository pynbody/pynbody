import pynbody
import subprocess
import os.path
import glob
import shutil
import stat

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