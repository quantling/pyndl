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

pkg = __import__('pyndl')

author =  pkg.__author__
email = pkg.__author_email__

version = pkg.__version__
classifiers = pkg.__classifiers__

description = pkg.__description__

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
    version=version,
    license='MIT',
    description=description,
    long_description=open('README.rst').read(),
    author=author,
    author_email=email,
    url='https://github.com/quantling/pyndl',
    classifiers=classifiers,
    platforms='Linux',
    packages=['pyndl'],
    install_requires=load_requirements('requirements.txt'),
    extras_require={
        'tests': [
            'pylint',
            'pytest',
            'pycodestyle'],
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme',
            'numpydoc',
            'easydev==0.9.35']},
    ext_modules=ext_modules,
    cmdclass=cmdclass
)
