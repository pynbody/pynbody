import pynbody
import subprocess
import os.path
import glob
import shutil
import stat
from pynbody.halo.adaptahop import AdaptaHOPCatalogue

def test_load_adaptahop_catalogue():
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = AdaptaHOPCatalogue(f, '/home/ccc/tmp/tree_bricks099')

    assert len(h) == 30875

def test_load():
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = AdaptaHOPCatalogue(f, '/home/ccc/tmp/tree_bricks099')

    h[1].properties

if __name__ == '__main__':
    f = pynbody.load('/home/ccc/Documents/prog/yt-data/output_00080/info_00080.txt') 
    h = AdaptaHOPCatalogue(f, '/home/ccc/tmp/tree_bricks099')