import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib

libraries=[ ]
extra_compile_args = ['-ftree-vectorizer-verbose=1', '-ftree-vectorize',
                      '-fno-omit-frame-pointer',
                      '-funroll-loops',
                      '-fprefetch-loop-arrays',
                      '-fstrict-aliasing',
                      '-std=c99',
                      '-Wall']

extra_link_args = []

app_dir = 'pynbody'
extra_files = ['default_config.ini', 'sph_image.c']

incdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
kdmain = Extension('pynbody/kdmain',
            sources = ['pynbody/kdmain.c', 'pynbody/kd.c', 'pynbody/smooth.c'],
		    include_dirs=[incdir],
            undef_macros=['DEBUG'],
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)

dist = setup(name = 'pyNBody',
      author = '',
      author_email = '',
      version = '1.0',
      description = '',
      package_dir = {'pynbody': ''},
      packages = ['pynbody/', 'pynbody/analysis', 'pynbody/bc_modules', 'pynbody/plot' ],
      ext_modules = [kdmain])

if dist.have_run.get('install'):
    install = dist.get_command_obj('install')

    # Copy textfiles in site-package directory
    for file in extra_files:
        install.copy_file(os.path.join(app_dir,file), os.path.join(install.install_lib, app_dir))

