import os
import shutil
import sys
from distutils.core import Distribution, Extension

from Cython.Build import build_ext, cythonize
import numpy

cython_dir = "pyndl"

ndl_parallel = Extension("pyndl.ndl_parallel",
                         ["pyndl/ndl_parallel.pyx"],
                         include_dirs=[numpy.get_include()])
ndl_openmp = Extension("pyndl.ndl_openmp",
                       ["pyndl/ndl_openmp.pyx"],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'],
                       include_dirs=[numpy.get_include()])
corr_parallel = Extension("pyndl.correlation_openmp",
                          ["pyndl/correlation_openmp.pyx"],
                          extra_compile_args=['-fopenmp'],
                          extra_link_args=['-fopenmp'],
                          include_dirs=[numpy.get_include()])

extensions = []
include_paths = []
if sys.platform.startswith('linux'):
    extensions = [ndl_parallel, ndl_openmp, corr_parallel]
    include_paths = [cython_dir, cython_dir, cython_dir]
elif sys.platform.startswith('win32'):
    extensions = [ndl_parallel] # skip openmp installation on windows for now
    include_paths = [cython_dir]
elif sys.platform.startswith('darwin'):
    extensions = [ndl_parallel]  # skip openmp installation on macos for now
    include_paths = [cython_dir]

ext_modules = cythonize(extensions, include_path=include_paths)
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)

