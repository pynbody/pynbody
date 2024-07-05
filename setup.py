import codecs
import os
import platform
import subprocess
from os import path

import numpy as np
from Cython.Build import build_ext
from Cython.Compiler.Options import get_directive_defaults
from setuptools import Extension, setup

get_directive_defaults()['language_level'] = 3

def is_macos():
    return platform.system() == 'Darwin'

def get_xcode_version():
    result = subprocess.run(['xcodebuild', '-version'], capture_output=True, text=True)
    version_line = result.stdout.split('\n')[0]
    version = version_line.split(' ')[1]
    return version

def xcode_fix_needed():
    if is_macos() and int(get_xcode_version().split('.')[0]) >= 15:
        return True
    else:
        return False

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")



# Support for compiling without OpenMP has been removed, for now, due to the spiralling
# complexities of making it work.
#
# Hopefully the availability of wheels for MacOS systems will prevent too many users suffering
openmp_module_source = "openmp/openmp_real"
openmp_args = ['-fopenmp']

ext_modules = []
libraries=[ ]
extra_compile_args = ['-ftree-vectorize',
                      '-fno-omit-frame-pointer',
                      '-funroll-loops',
                      '-fprefetch-loop-arrays',
                      '-fstrict-aliasing',
                      '-fno-expensive-optimizations', #<-- for arm64 gcc
                      '-g', '-std=c++14']

# note on -fno-expensive-optimizations:
# This is needed for arm64 gcc, which otherwise gets wrong results for a small number of particles
# in the kdtree_test.py::test_smooth_wendlandC2 test. It's unclear why; quite possibly there is
# a subtle bug in the code exposed by these optimizations, but it is such a vague optimization
# that it's hard to know what it is. The actual routine affected is smBallGather, but for some reason
# its impact only shows up with the Wendland kernel.

extra_link_args = openmp_args + ['-std=c++14']

if xcode_fix_needed():
    # workaround for XCode bug FB13097713
    # https://developer.apple.com/documentation/xcode-release-notes/xcode-15-release-notes#Linking
    extra_link_args += ['-Wl,-ld_classic']

incdir = [np.get_include()]

kdmain = Extension('pynbody.kdtree.kdmain',
                   sources = ['pynbody/kdtree/kdmain.cpp', 'pynbody/kdtree/kd.cpp',
                              'pynbody/kdtree/smooth.cpp'],
                   include_dirs=incdir,
                   undef_macros=['DEBUG'],

                   libraries=libraries,
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

ext_modules.append(kdmain)

gravity = Extension('pynbody.gravity._gravity',
                        sources = ["pynbody/gravity/_gravity.pyx"],
                        include_dirs=incdir,
                        extra_compile_args=openmp_args,
                        extra_link_args=extra_link_args)

omp_commands = Extension('pynbody.openmp',
                        sources = ["pynbody/"+openmp_module_source+".pyx"],
                        include_dirs=incdir,
                        extra_compile_args=openmp_args,
                        extra_link_args=extra_link_args)

chunkscan = Extension('pynbody.chunk.scan',
                  sources=['pynbody/chunk/scan.pyx'],
                  include_dirs=incdir)

sph_render = Extension('pynbody.sph._render',
                  sources=['pynbody/sph/_render.pyx'],
                  include_dirs=incdir)

halo_pyx = Extension('pynbody.analysis._com',
                     sources=['pynbody/analysis/_com.pyx'],
                     include_dirs=incdir,
                     extra_compile_args=openmp_args,
                     extra_link_args=extra_link_args)

bridge_pyx = Extension('pynbody.bridge._bridge',
                     sources=['pynbody/bridge/_bridge.pyx'],
                     include_dirs=incdir)

util_pyx = Extension('pynbody.util._util',
                     sources=['pynbody/util/_util.pyx'],
                     include_dirs=incdir,
                     extra_compile_args=openmp_args,
                     extra_link_args=extra_link_args)

filt_geom_pyx = Extension('pynbody.filt.geometry_selection',
                     sources=['pynbody/filt/geometry_selection.pyx'],
                     include_dirs=incdir,
                     extra_compile_args=openmp_args,
                     extra_link_args=extra_link_args,
                     language='c++')

cython_fortran_file = Extension('pynbody.extern._cython_fortran_utils',
                                sources=['pynbody/extern/_cython_fortran_utils.pyx'],
                                include_dirs=incdir)


interpolate3d_pyx = Extension('pynbody.analysis._interpolate3d',
                              sources = ['pynbody/analysis/_interpolate3d.pyx'],
                              include_dirs=incdir,
                              extra_compile_args=openmp_args,
                              extra_link_args=openmp_args)


ext_modules += [gravity, chunkscan, sph_render, halo_pyx, bridge_pyx, util_pyx, filt_geom_pyx,
                cython_fortran_file, interpolate3d_pyx, omp_commands]

install_requires = [
    'cython>=0.20',
    'h5py>=3.0.0',
    'matplotlib>=3.0.0',
    'numpy>=1.21.6',
    'scipy>=1.0.0',
    'posix-ipc>=1.1.0'
]

tests_require = [
    'pytest','pandas','camb','extinction'
]

docs_require = [
    'ipython>=3',
    'Sphinx>=7',
    'sphinx-book-theme',
    'sphinx-copybutton',
    'numpydoc',
    'extinction',
    'nbsphinx',
    'camb'
],

extras_require = {
    'tests': tests_require,
    'docs': docs_require,
}

extras_require['all'] = []
for name, reqs in extras_require.items():
    extras_require['all'].extend(reqs)



this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as snap:
    long_description = snap.read()

setup(name = 'pynbody',
      author = 'The pynbody team',
      author_email = 'pynbody@googlegroups.com',
      version = get_version("pynbody/__init__.py"),
      description = 'Light-weight astronomical N-body/SPH analysis for python',
      url = 'https://github.com/pynbody/pynbody/releases',
      package_dir = {'pynbody/': ''},
      packages = ['pynbody', 'pynbody/analysis', 'pynbody/array',
                  'pynbody/plot', 'pynbody/gravity', 'pynbody/chunk', 'pynbody/filt', 'pynbody/sph',
                  'pynbody/snapshot', 'pynbody/bridge', 'pynbody/halo', 'pynbody/halo/details',
                  'pynbody/extern', 'pynbody/kdtree', 'pynbody/test_utils', 'pynbody/util'],
      package_data={'pynbody': ['default_config.ini'],
                    'pynbody/analysis': ['cmdlum.npz', 'default_ssp.txt', 'lsst_ssp.txt',
                                         'h1.hdf5',
                                         'ionfracs.npz',
                                         'CAMB_WMAP7', 'CAMB_Planck18',
                                         'cambtemplate.ini'],
                    'pynbody/plot': ['tollerud2008mw']},
      ext_modules = ext_modules,
      classifiers = ["Development Status :: 4 - Beta",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                     "Programming Language :: Python :: 3",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Topic :: Scientific/Engineering :: Visualization"],
      cmdclass={'build_ext': build_ext},
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      python_requires='>=3.10',
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
