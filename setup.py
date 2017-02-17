from setuptools import setup, Extension

# While setup, this file will be called twice:
# One time, to read the dependencies,
# and after their installation.
try:
    from Cython.Distutils import build_ext
    import numpy
except ImportError as e:
    use_deps = False
else:
    use_deps = True


def load_requirements(fn):
    """Read a requirements file and create a list that can be used in setup."""
    with open(fn, 'r') as f:
        return [x.rstrip() for x in list(f) if x and not x.startswith('#')]


if use_deps:
    ext_modules = [
        Extension(
            "pyndl.ndl_parallel",
            ["pyndl/ndl_parallel.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()]
        )
    ]
    cmdclass = {'build_ext': build_ext}
else:
    ext_modules = []
    cmdclass = {}

setup(
    name='pyndl',
    version='0.1dev',
    description=('Naive discriminative learning implements learning and '
                 'classification models based on the Rescorla-Wagner equations.'),
    long_description=open('README.rst').read(),
    author=('David-Elias Künstle, Lennard Schneider, '
            'Konstantin Sering, Marc Weitz'),
    author_email='konstantin.sering@uni-tuebingen.de',
    url='http://www.sfs.uni-tuebingen.de/en/ql.html',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    platforms='Linux',
    packages=['pyndl'],
    install_requires=load_requirements('requirements.txt'),
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
