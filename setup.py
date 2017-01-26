from setuptools import setup, Extension

# While setup, this file will be called twice:
# One time, to read the dependencies,
# and after their installation.
try:
    from Cython.Build import cythonize
    import numpy
except ImportError as e:
    use_deps = False
else:
    use_deps = True


def load_requirements(fn):
    """ Read a requirements file and create a list that can be used in setup. """
    with open(fn, 'r') as f:
        return [x.rstrip() for x in list(f) if x and not x.startswith('#')]


if use_deps:
    ext_modules = cythonize([
        Extension(
            "ndl_parallel",
            ["pyndl/ndl_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
        Extension(
            "ndl_c",
            ["pyndl/ndl_c.pyx"]
        )
    ])
    include_dirs = [numpy.get_include()]
else:
    ext_modules = []
    include_dirs = []

setup(
    name='pyndl',
    version='0.1',
    packages=['pyndl'],
    install_requires=load_requirements('requirements.txt'),
    ext_modules=ext_modules,
    include_dirs=include_dirs
)
