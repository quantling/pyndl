Installation
============

Supported systems and versions
------------------------------

.. image:: https://img.shields.io/travis/quantling/pyndl/main.svg?maxAge=3600&label=Linux
    :target: https://travis-ci.org/quantling/pyndl?branch=main

.. image:: https://img.shields.io/pypi/pyversions/pyndl.svg
    :target: https://pypi.python.org/pypi/pyndl/

*pyndl* currently is only tested and mainly used on 64-bit Linux systems.
However, it is possible to install it on other operating systems, but be aware
that some functionality might not work or will not work as intended. Therefore
be extra careful and run the test suite after installing it on a non Linux
system.

.. note::

    If you face problems with installing *pyndl* with `pip`, it might be
    helpful to use `Minicoda <https://conda.io/miniconda.html>`_ to install the
    following dependencies::

        conda install numpy cython pandas xarray netCDF4 numpydoc pip

    The reason behind this is that during the installation process of *pyndl*
    Cython extension need to be installed, if no pre-compiled wheel could be
    found for your operating system and architecture. To compile Cython
    extensions some further steps need to be done, which is described in the
    `Cython documentation <https://cython.readthedocs.io>`_ . These steps depend
    on your operating system. Installing Cython with `conda install cython`
    should add all the necessary additional programs and files and no further
    steps are needed.


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
On Windows installing ``pyndl`` should work with ``pip`` as well. Execute in the PowerShell:

.. code:: bash

    pip install --user pyndl

If this fails for some reason you can try to install ``pyndl`` with `poetry
<https://python-poetry.org/>`_. First you need to install ``poetry`` and
``git``. After both programs are installed properly you should be able install
``pyndl`` with the following commands in the PowerShell:

.. code:: bash

   git clone https://github.com/quantling/pyndl
   cd pyndl
   poetry install

Test the installation with:

.. code:: bash

   poetry run pytest --on-linux


.. note::

    ``pyndl`` on Windows and MacOS X comes without OpenMP support and therefore
    some functionality is not available.

