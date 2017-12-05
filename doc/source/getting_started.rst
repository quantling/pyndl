Getting Started
===============

These instructions will get you a copy of the ``pyndl`` package on your local
machine. If you only want to use ``pyndl`` as a python package use ``pip3`` in
order to install it into your python3 environment. If you want to inspect and
change the code download and install it via ``git clone`` and ``python3
setup.py``. For details see below.


Prerequisites
-------------
You need python 3.4 or newer and git installed on your machine. We recommend to
install `Minicoda <https://conda.io/miniconda.html>`_ before installing ``pyndl``
or to create a virtualenv within your personal folder.

Development
^^^^^^^^^^^
If you want to develop ``pyndl`` you should additionally install:

.. code:: bash

   pip3 install --user tox pylint pytest pycodestyle sphinx


Installation
------------

Currently, we ar only supporting the installation of ``pyndl`` on Linux systems.
However, feel free to try installing it on other operating systems, but be aware
that some functionality might not work or will not work as intended.

Linux
^^^^^

If you want to install pyndl on Linux and only want to use the package run you
can easily install ``pyndl`` from pypi with:

.. code:: bash

    pip3 install pyndl --user

If you want to inspect and change the source code as well as running tests and
having local documentation, clone the repository and install the package in
'development' mode by running

.. code:: bash

    git clone https://github.com/quantling/pyndl.git
    cd pyndl
    python3 setup.py develop --user


Mac OS X
^^^^^^^^
gcc/g++ might be outdated as xcode provides 4.X, while 6.3 is needed. Therefore,
it might be necesarry to update gcc first, before installing pyndl.

1. for safe-guarding redo the xcode install in the Terminal:

.. code:: bash

        xcode-select --install

2. download gcc from `Mac OSX High Performance Computing <http://prdownloads.sourceforge.net/hpc/gcc-6.3-bin.tar.gz>`_
then run these commands in Terminal:

.. code:: bash

        gunzip gcc-6.X-bin.tar.gz
        sudo tar -xvf gcc-6.X-bin.tar -C /

3. finally, install pyndl:

.. code:: bash

        pip install pyndl

Hopefully, you have successfully installed pyndl on your OS X Machine.

.. warning::

    Please be aware that we currently offer no support for OS X and can therefore
    not provide, whether a safe installation or usage on OS X is possible.
