import codecs
import distutils
import os
import shutil
import subprocess
import sys
import tempfile
from os import path

import numpy.distutils.misc_util
from Cython.Build import build_ext
from Cython.Compiler.Options import get_directive_defaults
from setuptools import Extension, setup

get_directive_defaults()['language_level'] = 3


def check_for_pthread():
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              distutils.sysconfig.get_config_var('CC'))

    # make sure to use just the compiler name without flags
    compiler = compiler.split()[0]

    filename = r'test.c'
    with open(filename,'w') as f :
        f.write(
        "#include <pthread.h>\n"
        "#include <stdio.h>\n"
        "int main() {\n"
        "}"
        )

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, filename],
                                        stdout=fnull, stderr=fnull)
    except OSError :
        exit_code = 1


    # Clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    return (exit_code==0)

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


have_pthread = check_for_pthread()


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
                      '-g']

if have_pthread:
    extra_compile_args.append('-DKDT_THREADING')

extra_link_args = []

incdir = numpy.distutils.misc_util.get_numpy_include_dirs()

kdmain = Extension('pynbody.sph.kdmain',
                   sources = ['pynbody/sph/kdmain.cpp', 'pynbody/sph/kd.cpp',
                              'pynbody/sph/smooth.cpp'],
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
                        extra_link_args=openmp_args)

omp_commands = Extension('pynbody.openmp',
                        sources = ["pynbody/"+openmp_module_source+".pyx"],
                        include_dirs=incdir,
                        extra_compile_args=openmp_args,
                        extra_link_args=openmp_args)

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
                     extra_link_args=openmp_args)

bridge_pyx = Extension('pynbody.bridge._bridge',
                     sources=['pynbody/bridge/_bridge.pyx'],
                     include_dirs=incdir)

util_pyx = Extension('pynbody._util',
                     sources=['pynbody/_util.pyx'],
                     include_dirs=incdir,
                     extra_compile_args=openmp_args,
                     extra_link_args=openmp_args)

cython_fortran_file = Extension('pynbody.extern._cython_fortran_utils',
                                sources=['pynbody/extern/_cython_fortran_utils.pyx'],
                                include_dirs=incdir)

cosmology_time = Extension('pynbody.analysis._cosmology_time',
                           sources=['pynbody/analysis/_cosmology_time.pyx'],
                           include_dirs=incdir)

interpolate3d_pyx = Extension('pynbody.analysis._interpolate3d',
                              sources = ['pynbody/analysis/_interpolate3d.pyx'],
                              include_dirs=incdir,
                              extra_compile_args=openmp_args,
                              extra_link_args=openmp_args)


ext_modules += [gravity, chunkscan, sph_render, halo_pyx, bridge_pyx, util_pyx,
                cython_fortran_file, cosmology_time, interpolate3d_pyx, omp_commands]

install_requires = [
    'cython>=0.20',
    'h5py>=2.10.0',
    'matplotlib>=3.0.0',
    'numpy>=1.21.6',
    'posix_ipc>=0.8',
    'scipy>=1.0.0'
]

tests_require = [
    'pytest','pandas'
]

docs_require = [
    'ipython>=3',
    'Sphinx==1.6.*',
    'sphinx-bootstrap-theme',
],

extras_require = {
    'tests': tests_require,
    'docs': docs_require,
}

extras_require['all'] = []
for name, reqs in extras_require.items():
    extras_require['all'].extend(reqs)



this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name = 'pynbody',
      author = 'The pynbody team',
      author_email = 'pynbody@googlegroups.com',
      version = get_version("pynbody/__init__.py"),
      description = 'Light-weight astronomical N-body/SPH analysis for python',
      url = 'https://github.com/pynbody/pynbody/releases',
      package_dir = {'pynbody/': ''},
      packages = ['pynbody', 'pynbody/analysis', 'pynbody/bc_modules',
                  'pynbody/plot', 'pynbody/gravity', 'pynbody/chunk', 'pynbody/sph',
                  'pynbody/snapshot', 'pynbody/bridge', 'pynbody/halo', 'pynbody/extern'],
      package_data={'pynbody': ['default_config.ini'],
                    'pynbody/analysis': ['cmdlum.npz',
                                         'h1.hdf5',
                                         'ionfracs.npz',
                                         'CAMB_WMAP7',
                                         'cambtemplate.ini'],
                    'pynbody/plot': ['tollerud2008mw']},
      ext_modules = ext_modules,
      classifiers = ["Development Status :: 5 - Production/Stable",
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
      python_requires='>=3.8',
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
