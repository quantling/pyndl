from setuptools import setup, Extension


try:
    from Cython.Build import cythonize
except ImportError as e:
    use_cython = False
else:
    use_cython = True


def load_requirements(fn):
    """ Read a requirements file and create a list that can be used in setup. """
    with open(fn, 'r') as f:
        return [x.rstrip() for x in list(f) if x and not x.startswith('#')]


ext_modules = []
if use_cython:
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
else:
    pass

setup(
    name='pyndl',
    version='0.1',
    packages=['pyndl'],
    install_requires=load_requirements('requirements.txt'),
    ext_modules=ext_modules
)
