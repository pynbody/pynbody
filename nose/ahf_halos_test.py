import pynbody
import subprocess

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