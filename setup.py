from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# bootstrap numpy
# https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


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


# by giving ``cython`` as ``install_requires`` this will be ``cythonized``
# automagically
ext_modules = [
    Extension(
        "pyndl.ndl_parallel",
        ["pyndl/ndl_parallel.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]


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
    setup_requires=['numpy', 'cython'],
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
    cmdclass={'build_ext': build_ext}
)
