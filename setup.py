import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

import numpy
import numpy.distutils.misc_util

libraries=[ ]
extra_compile_args = ['-ftree-vectorizer-verbose=1', '-ftree-vectorize',
                      '-fno-omit-frame-pointer',
                      '-funroll-loops',
                      '-fprefetch-loop-arrays',
                      '-fstrict-aliasing',
                      '-std=c99',
                      '-Wall',
                      '-O0']

extra_link_args = []

incdir = numpy.distutils.misc_util.get_numpy_include_dirs()

#os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
kdmain = Extension('pynbody/kdmain',
                   sources = ['pynbody/kdmain.c', 'pynbody/kd.c', 
                              'pynbody/smooth.c'],
                   include_dirs=incdir,
                   undef_macros=['DEBUG'],
                   libraries=libraries,
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

gravity = Extension('pynbody/grav',
                   sources = ['pynbody/gravity/main.c', 
                              'pynbody/gravity/serialtree.c',
                              'pynbody/gravity/walk.c',
                              'pynbody/gravity/grav.c',
                              'pynbody/gravity/moments.c'],
                   include_dirs=incdir,
                   undef_macros=['DEBUG'],
                   libraries=libraries,
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

dist = setup(name = 'pynbody',
             author = '',
             author_email = '',
             version = '0.13alpha',
             description = '',
             package_dir = {'pynbody/': ''},
             packages = ['pynbody', 'pynbody/analysis', 'pynbody/bc_modules', 
                         'pynbody/plot', 'pynbody/gravity' ],
# treat weave .c files like data files since weave takes
# care of their compilation for now
# could make a separate extension for them in future
             package_data={'pynbody': ['default_config.ini', 
                                       'sph_image.c'],
                           'pynbody/analysis': ['cmdlum.npz',
                                                'ionfracs.npz',
                                                'interpolate.c',
                                                'interpolate3d.c',
                                                'CAMB_WMAP7'],
                           'pynbody/plot': ['tollerud2008mw'],
                           'pynbody/gravity': ['direct.c']},
             ext_modules = [kdmain, gravity]
      )

#if dist.have_run.get('install'):
#    install = dist.get_command_obj('install')

