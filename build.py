from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import sys


# bootstrap numpy
# https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        #__builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


ndl_parallel = Extension("pyndl.ndl_parallel", ["pyndl/ndl_parallel.pyx"])
ndl_openmp = Extension("pyndl.ndl_openmp", ["pyndl/ndl_openmp.pyx"],
                       extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
corr_parallel = Extension("pyndl.correlation_openmp", ["pyndl/correlation_openmp.pyx"],
                       extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
# by giving ``cython`` as ``install_requires`` this will be ``cythonized``
# automagically

ext_modules = []
if sys.platform.startswith('linux'):
    ext_modules = [ndl_parallel, ndl_openmp, corr_parallel]
elif sys.platform.startswith('win32'):
    ext_modules = [ndl_parallel] # skip openmp installation on windows for now
elif sys.platform.startswith('darwin'):
    ext_modules = [ndl_parallel]  # skip openmp installation on macos for now


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': ext_modules,
        'cmdclass': {
            'build_ext': build_ext
        }
    })
