import pynbody
import subprocess
import os.path
import glob
import shutil
import stat
from pynbody.halo.adaptahop import AdaptaHOPCatalogue

def test_load_adaptahop_catalogue():
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = f.halos(read_contamination=True)
    assert len(h) == 151

def test_load_one_halo():
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = f.halos(read_contamination=True)
    h[1]

if __name__ == '__main__':
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = AdaptaHOPCatalogue(f, '/home/ccc/tmp/tree_bricks099')