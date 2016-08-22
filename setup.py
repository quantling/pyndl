from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = 'cyndl'
        ext_modules = cythonize('ndl_c.pyx')
)
