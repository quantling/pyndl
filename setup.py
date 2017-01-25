from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "ndl_parallel",
        ["ndl_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        "ndl_c",
        ["ndl_c.pyx"]
    )
]

setup(
    # name = 'cyndl',
    ext_modules=cythonize(ext_modules)
)
