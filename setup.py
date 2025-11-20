import os
import platform
import subprocess

import numpy as np
from Cython.Build import build_ext
from Cython.Compiler.Options import get_directive_defaults
from setuptools import Extension, setup

get_directive_defaults()['language_level'] = 3

def is_macos():
    return platform.system() == 'Darwin'

def is_windows():
    return platform.system() == 'Windows'

def get_xcode_version():
    result = subprocess.run(['pkgutil', '--pkg-info=com.apple.pkg.CLTools_Executables'], capture_output=True, text=True)
    try:
        version_line = result.stdout.split('\n')[1]
        version = version_line.split(' ')[1]
    except IndexError:
        return 0 # looks like xcode-cltools not installed? try to proceed anyway
    return version

def xcode_fix_needed():
    if is_macos() and int(get_xcode_version().split('.')[0]) >= 15:
        return True
    else:
        return False


# Support for compiling without OpenMP has been removed, for now, due to the spiralling
# complexities of making it work.
#
# Hopefully the availability of wheels for MacOS systems will prevent too many users suffering
openmp_module_source = "openmp/openmp_real"

# Platform-specific compiler settings
if is_windows():
    # MSVC compiler flags
    openmp_args = ['/openmp']
    extra_compile_args = ['/O2', '/std:c++14']
    extra_link_args = ['/openmp']
else:
    use_static_openmp = os.environ.get('OPENMP_STATIC', '0') == '1'

    # GCC/Clang compiler flags
    openmp_args = ['-fopenmp']

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

ext_modules = []
libraries=[ ]

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
                  sources=['pynbody/sph/_render.pyx', 'pynbody/sph/healpix.c'],
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
                     sources=['pynbody/util/_util.pyx', 'pynbody/sph/healpix.c'],
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


ext_modules += [gravity, chunkscan, sph_render, halo_pyx, bridge_pyx, util_pyx, filt_geom_pyx,
                cython_fortran_file, omp_commands]

setup(ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext})
