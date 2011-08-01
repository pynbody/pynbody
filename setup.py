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

dist = setup(name = 'pynbody',
             author = '',
             author_email = '',
             version = '0.22beta',
             description = '',
             package_dir = {'pynbody': ''},
             packages = ['pynbody/', 'pynbody/analysis', 'pynbody/bc_modules', 
                         'pynbody/plot' ],
             ext_modules = [kdmain],
# treat weave .c files like data files since weave takes
# care of their compilation for now
# could make a separate extension for them in future
             data_files = [('pynbody',['pynbody/default_config.ini', 
                                       'pynbody/sph_image.c']), 
                           ('pynbody/analysis',['pynbody/analysis/cmdlum.npz',
                                                'pynbody/analysis/interpolate.c']), 
                           ('pynbody/plot',['pynbody/plot/tollerud2008mw'])],
             scripts = ['scripts/doall.py']
      )

#if dist.have_run.get('install'):
#    install = dist.get_command_obj('install')

