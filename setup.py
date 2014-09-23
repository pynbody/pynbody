import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

import numpy
import numpy.distutils.misc_util
import glob
import tempfile
import subprocess
import shutil
import sys


# Patch the sdist command to ensure both versions of the openmp module are
# included with source distributions
#
# Solution from http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code

from distutils.command.sdist import sdist as _sdist

class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        for f in glob.glob("pynbody/openmp/*.pyx"):
            cythonize([f])
        _sdist.run(self)

cmdclass = {'sdist':sdist}


def check_for_openmp():
    """Check  whether the default compiler supports OpenMP.

    This routine is adapted from yt, thanks to Nathan
    Goldbaum. See https://github.com/pynbody/pynbody/issues/124"""
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    # Get compiler invocation
    compiler = os.environ.get('CC',
                              distutils.sysconfig.get_config_var('CC'))

    # make sure to use just the compiler name without flags
    compiler = compiler.split()[0]

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
    filename = r'test.c'
    with open(filename,'w') as f :
        f.write(
        "#include <omp.h>\n"
        "#include <stdio.h>\n"
        "int main() {\n"
        "#pragma omp parallel\n"
        "printf(\"Hello from thread %d, nthreads %d\\n\", omp_get_thread_num(), omp_get_num_threads());\n"
        "}"
        )

    try:
        with open(os.devnull, 'w') as fnull:
            exit_code = subprocess.call([compiler, '-fopenmp', filename],
                                        stdout=fnull, stderr=fnull)
    except OSError :
        exit_code = 1

    # Clean up
    os.chdir(curdir)
    shutil.rmtree(tmpdir)

    if exit_code == 0:
        return True
    else:
        return False

cython_version = None
try :
    import cython

    # check that cython version is > 0.20
    cython_version = cython.__version__
    if float(cython_version.partition(".")[2][:2]) < 20 :
	print "yikes! error importing correct cython", float(cython_version.partition(".")[2][:2])
        raise ImportError
    from Cython.Distutils import build_ext
    build_cython = True
    cmdclass['build_ext']=build_ext
except:
    build_cython = False

import distutils.command.build_py

try :
    cmdclass['build_py'] =  distutils.command.build_py.build_py_2to3
except AttributeError:
    cmdclass['build_py'] =  distutils.command.build_py.build_py

have_openmp = check_for_openmp()

if have_openmp :
    openmp_module_source = "openmp/openmp_real"
    openmp_args = ['-fopenmp']
else :
    openmp_module_source = "openmp/openmp_null"
    openmp_args = ['']

ext_modules = []
libraries=[ ]
extra_compile_args = ['-ftree-vectorize',
                      '-fno-omit-frame-pointer',
                      '-funroll-loops',
                      '-fprefetch-loop-arrays',
                      '-fstrict-aliasing',
                      '-std=c99',
                      '-Wall',
                      '-O0',
                      '-g']

if sys.version_info[0:2]==(3,4) :
    # this fixes the following bug with the python 3.4 build:
    # http://bugs.python.org/issue21121
    extra_compile_args.append("-Wno-error=declaration-after-statement")


extra_link_args = []

incdir = numpy.distutils.misc_util.get_numpy_include_dirs()

kdmain = Extension('pynbody/sph/kdmain',
                   sources = ['pynbody/sph/kdmain.c', 'pynbody/sph/kd.c',
                              'pynbody/sph/smooth.c'],
                   include_dirs=incdir,
                   undef_macros=['DEBUG'],
                   libraries=libraries,
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

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

interpolate3d_pyx = Extension('pynbody.analysis._interpolate3d',
                              sources = ['pynbody/analysis/_interpolate3d.pyx'],
                              include_dirs=incdir,
                              extra_compile_args=openmp_args,
                              extra_link_args=openmp_args)


ext_modules+=[kdmain,gravity,chunkscan,sph_render,halo_pyx,bridge_pyx, util_pyx,interpolate3d_pyx, omp_commands]

if not build_cython :
    for mod in ext_modules :
        mod.sources = list(map(lambda source: source.replace(".pyx",".c"),
                           mod.sources))
        for src in mod.sources:
            if not os.path.isfile(src):
                print ("""
You are attempting to install pynbody without cython. Unfortunately
this package does not include the generated .c files that are required
to do so.

You have two options. Either:

 1. Get a 'release' version of pynbody from
    https://github.com/pynbody/pynbody/releases

or

 2. Install Cython, at least version 0.20 and preferably 0.21 or higher.
    This can normally be accomplished by typing

    pip install --upgrade cython.


If you already did one of the above, you've encountered a bug. Please
open an issue on github to let us know. The missing file is {0}
and the detected cython version is {1}.
""".format(src,cython_version))

                sys.exit(1)

dist = setup(name = 'pynbody',
             install_requires='numpy>=1.5',
             author = 'The pynbody team',
             author_email = 'pynbody@googlegroups.com',
             version = '0.30',
             description = 'Light-weight astronomical N-body/SPH analysis for python',
             url = 'https://github.com/pynbody/pynbody/releases',
             package_dir = {'pynbody/': ''},
             packages = ['pynbody', 'pynbody/analysis', 'pynbody/bc_modules',
                         'pynbody/plot', 'pynbody/gravity', 'pynbody/chunk', 'pynbody/sph',
                         'pynbody/snapshot', 'pynbody/bridge' ],
# treat weave .c files like data files since weave takes
# care of their compilation for now
# could make a separate extension for them in future
             package_data={'pynbody': ['default_config.ini'],
                           'pynbody/sph': ['sph_image.c','sph_to_grid.c',
                                        'sph_spectra.c'],
                           'pynbody/analysis': ['cmdlum.npz',
                                                'ionfracs.npz',
                                                'interpolate.c',
                                                'interpolate3d.c',
                                                'com.c',
                                                'CAMB_WMAP7',
                                                'cambtemplate.ini'],
                           'pynbody/plot': ['tollerud2008mw'],
                           'pynbody/gravity': ['direct.c']},
             ext_modules = ext_modules,
             cmdclass = cmdclass,
             classifiers = ["Development Status :: 4 - Beta",
                            "Intended Audience :: Developers",
                            "Intended Audience :: Science/Research",
                            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                            "Programming Language :: Python :: 2",
                            "Programming Language :: Python :: 3",
                            "Topic :: Scientific/Engineering :: Astronomy",
                            "Topic :: Scientific/Engineering :: Visualization"]

      )

#if dist.have_run.get('install'):
#    install = dist.get_command_obj('install')
