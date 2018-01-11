Installation
============

Supported systems and versions
------------------------------

.. image:: https://img.shields.io/travis/quantling/pyndl/master.svg?maxAge=3600&label=Linux
    :target: https://travis-ci.org/quantling/pyndl?branch=master

.. image:: https://img.shields.io/pypi/pyversions/pyndl.svg
    :target: https://pypi.python.org/pypi/pyndl/

*pyndl* currently only supports installation on Linux systems. However, it is
possible to install it on other operating systems, but be aware
that some functionality might not work or will not work as intended.

.. note::

  We recommend to install `Minicoda <https://conda.io/miniconda.html>`_ before
  installing *pyndl* or to create a virtualenv within your personal folder.


Linux
-----

If you want to install *pyndl* on Linux the easiest way is to install it
from `pypi <https://pypi.python.org/pypi>`_ with:

.. code:: bash

    pip install pyndl --user

MacOS
-----

If you want to install *pyndl* on MacOS you can also install it from
`pypi <https://pypi.python.org/pypi>`_. However, gcc/g++ might be outdated as
xcode provides 4.X, while 6.3 is needed. Therefore, it might be necessary to
update gcc first, before installing *pyndl*:

1. for safe-guarding redo the xcode install in the Terminal:

.. code:: bash

        xcode-select --install

2. download gcc from `Mac OSX High Performance Computing
<http://prdownloads.sourceforge.net/hpc/gcc-6.3-bin.tar.gz>`_
then run these commands in Terminal:

.. code:: bash

        gunzip gcc-6.X-bin.tar.gz
        sudo tar -xvf gcc-6.X-bin.tar -C /

3. finally, install pyndl:

.. code:: bash

    pip install pyndl --user

.. warning::

    This procedure is experimental and might not work. As long as we do not
    actively support MacOS be aware that these installation instructions can
    fail or the installed package does not always works as intended!
