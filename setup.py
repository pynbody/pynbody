import os
from setuptools import setup, Extension

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

    # Attempt to compile a test script.
    # See http://openmp.org/wp/openmp-compilers/
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
        import multiprocessing, platform
        cpus = multiprocessing.cpu_count()
        if cpus>1:
            print ("""WARNING

OpenMP support is not available in your default C compiler, even though
your machine has more than one core available.

Some routines in pynbody are parallelized using OpenMP and these will
only run on one core with your current configuration.
""")
            if platform.uname()[0]=='Darwin':
                print ("""Since you are running on Mac OS, it's likely that the problem here
is Apple's Clang, which does not support OpenMP at all. The easiest
way to get around this is to download the latest version of gcc from
here: http://hpc.sourceforge.net. After downloading, just point the
CC environment variable to the real gcc and OpenMP support should
get enabled automatically. Something like this -

sudo tar -xzf /path/to/download.tar.gz /
export CC='/usr/local/bin/gcc'
python setup.py clean
python setup.py build

""")
            print ("""Continuing your build without OpenMP...\n""")

        return False

cython_version = None
try :
    import cython
    # check that cython version is > 0.21
    cython_version = cython.__version__
    if float(cython_version.partition(".")[2][:2]) < 21 :
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
have_pthread = check_for_pthread()

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
                      '-g']


if sys.version_info[0:2]==(3,4) :
    # this fixes the following bug with the python 3.4 build:
    # http://bugs.python.org/issue21121
    extra_compile_args.append("-Wno-error=declaration-after-statement")

if have_pthread:
    extra_compile_args.append('-DKDT_THREADING')

extra_link_args = []

incdir = numpy.distutils.misc_util.get_numpy_include_dirs()

kdmain = Extension('pynbody/sph/kdmain',
                   sources = ['pynbody/sph/kdmain.cpp', 'pynbody/sph/kd.cpp',
                              'pynbody/sph/smooth.cpp'],
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
You are attempting to install pynbody without a recent version of cython.
Unfortunately this pynbody package does not include the generated .c files that
are required to do so.

You have two options. Either:

 1. Get a 'release' version of pynbody from
    https://github.com/pynbody/pynbody/releases

or

 2. Install Cython version 0.21 or higher.

    This can normally be accomplished by typing

    pip install --upgrade cython.


If you already did one of the above, you've encountered a bug. Please
open an issue on github to let us know. The missing file is {0}
and the detected cython version is {1}.
""".format(src,cython_version))

                sys.exit(1)

install_requires = [
    'cython>=0.20',
    'h5py',
    'matplotlib',
    'numpy>=1.9.2',
    'pandas',
    'posix_ipc',
    'scipy',
]

tests_require = [
    'nose'
]

docs_require = [
    'ipython>=3',
    'Sphinx==1.6.*',
    'sphinx-bootstrap-theme',
],

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
}

extras_require['all'] = []
for name, reqs in extras_require.items():
    extras_require['all'].extend(reqs)

dist = setup(name = 'pynbody',
             author = 'The pynbody team',
             author_email = 'pynbody@googlegroups.com',
             version = '0.46',
             description = 'Light-weight astronomical N-body/SPH analysis for python',
             url = 'https://github.com/pynbody/pynbody/releases',
             package_dir = {'pynbody/': ''},
             packages = ['pynbody', 'pynbody/analysis', 'pynbody/bc_modules',
                         'pynbody/plot', 'pynbody/gravity', 'pynbody/chunk', 'pynbody/sph',
                         'pynbody/snapshot', 'pynbody/bridge', 'pynbody/halo' ],
             package_data={'pynbody': ['default_config.ini'],
                           'pynbody/analysis': ['cmdlum.npz',
                                                'h1.hdf5',
                                                'ionfracs.npz',
                                                'CAMB_WMAP7',
                                                'cambtemplate.ini'],
                           'pynbody/plot': ['tollerud2008mw']},
             ext_modules = ext_modules,
             cmdclass = cmdclass,
             classifiers = ["Development Status :: 5 - Production/Stable",
                            "Intended Audience :: Developers",
                            "Intended Audience :: Science/Research",
                            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                            "Programming Language :: Python :: 2",
                            "Programming Language :: Python :: 3",
                            "Topic :: Scientific/Engineering :: Astronomy",
                            "Topic :: Scientific/Engineering :: Visualization"],
             install_requires=install_requires,
             tests_require=tests_require,
             extras_require=extras_require,
      )

#if dist.have_run.get('install'):
#    install = dist.get_command_obj('install')
