import pynbody
import subprocess
import os.path
import glob
import shutil
import stat
from pynbody.halo.adaptahop import AdaptaHOPCatalogue

def test_load_adaptahop_catalogue():
    f = pynbody.load('/home/ccc/tmp/output_00099')
    h = f.halos()
    assert len(h) == h._headers['nhalos'] + h._headers['nsubs']

def test_load_one_halo():
    f = pynbody.load('/home/ccc/tmp/output_00099')
    h = f.halos()
    h[1]

if __name__ == '__main__':
    f = pynbody.load('/home/ccc/tmp/output_00099')
    h = f.halos()
