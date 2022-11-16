Installation
============

Supported systems and versions
------------------------------

.. image:: https://img.shields.io/travis/quantling/pyndl/master.svg?maxAge=3600&label=Linux
    :target: https://travis-ci.org/quantling/pyndl?branch=master

.. image:: https://img.shields.io/pypi/pyversions/pyndl.svg
    :target: https://pypi.python.org/pypi/pyndl/

*pyndl* currently is only tested and mainly used on 64-bit Linux systems.
However, it is possible to install it on other operating systems, but be aware
that some functionality might not work or will not work as intended. Therefore
be extra careful and run the test suite after installing it on a non Linux
system.

.. note::

    We recommend to install `Minicoda <https://conda.io/miniconda.html>`_
    before installing *pyndl* or to create a virtualenv within your personal
    folder.

    It is recommended, to install the following dependencies of `pyndl` through
    the `conda` command::

       conda install numpy cython pandas xarray netCDF4 numpydoc pip


.. note::

   During the installation process of *pyndl* Cython extension need to be
   installed, if no pre-compiled wheel could be found for your operating system
   and architecture. To compile Cython extensions some further steps need to be
   done, which is described in the `Cython documentation
   <https://cython.readthedocs.io>`_ . These steps depend on your operating
   system. Installing Cython with `conda install cython` should add all the
   necessary additional programs and files and no further steps are needed.


Linux
-----

If you want to install *pyndl* on Linux the easiest way is to install it
from `pypi <https://pypi.python.org/pypi>`_ with:

.. code:: bash

    pip install --user pyndl


MacOS
-----

If you want to install *pyndl* on MacOS you can also install it from
`pypi <https://pypi.python.org/pypi>`_. However, the installation will not have
`openmp` support. Sometimes an error is shown during the installation, but the
installations succeeds nonetheless. Before filing a bug report please check if
you can run the examples from the documentation.

Install *pyndl* with:

.. code:: bash

    pip install --user pyndl


Windows 10
----------

.. note::

    You might need to enable the ``bash`` within Windows 10 first to be able to
    follow the following instructions.

After installing Anaconda or Miniconda, first install the dependencies with the
``conda`` command in the bash or the Miniconda terminal:

.. code:: bash

    conda update conda
    conda install numpy cython pandas xarray netCDF4 numpydoc pip

After the installation of the dependencies finished successfully you should be
able to install ``pyndl`` with pip:

.. code:: bash

    pip install --user pyndl

.. warning::

    This procedure is experimental and might not work. As long as we do not
    actively support Windows 10 be aware that these installation instructions
    can fail or the installed package does not always works as intended!
