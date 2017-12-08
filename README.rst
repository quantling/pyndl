============
Readme pyndl
============

.. image:: https://travis-ci.org/quantling/pyndl.svg?branch=master
    :target: https://travis-ci.org/quantling/pyndl?branch=master

.. image:: https://landscape.io/github/quantling/pyndl/master/landscape.svg?style=flat
    :target: https://landscape.io/github/quantling/pyndl/master
    :alt: Code Health

.. image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. image:: https://img.shields.io/pypi/pyversions/pyndl.svg
    :target: https://pypi.python.org/pypi/pyndl/

.. image:: https://img.shields.io/github/license/quantling/pyndl.svg
    :target: https://github.com/quantling/pyndl/blob/master/LICENSE.txt

.. image:: https://zenodo.org/badge/80022085.svg
    :target: https://zenodo.org/badge/latestdoi/80022085


This python3 package is a collection of useful script in order to run tasks on
huge amounts of text file corpora. Especially, it allows to efficiently apply
the Rescorla-Wagner learning rule to these corpora.

.. warning::

    This package is still in alpha and there might be some API changes in the
    near future. If you have suggestings concerning the ``pyndl`` package,
    contact us under konstantin (dot) sering (ät) uni-tuebingen.de.

.. note::

    This package is not intended to be used under python2.


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
install Minicoda (https://conda.io/miniconda.html) before installing ``pyndl``
or to create a virtualenv within your personal folder.

Development
^^^^^^^^^^^
If you want to develop ``pyndl`` you should additionally install:

.. code:: bash

   pip3 install --user tox pylint pytest pycodestyle sphinx


Installing
----------
If you only want to use the package run you can install ``pyndl`` from pypi with:

.. code:: bash

    pip3 install --user pyndl

If you want to inspect and change the source code as well as running tests and
having local documentation, clone the repository and install the package in
'development' mode by running

.. code:: bash

    git clone https://github.com/quantling/pyndl.git
    cd pyndl
    python3 setup.py --user develop


Documentation and Examples
--------------------------
Documentation and examples can be found under
http://pyndl.readthedocs.io/en/latest/ or in the ``doc/`` folder after cloning
the repository.

If you have installed ``sphinx`` you should be able to build the documentation with:

.. code:: bash

   cd doc/
   make html

The entry point for the html documentation is ``doc/build/html/index.html``.


Running the tests
=================
If you have cloned ``pyndl`` from github, you can start the tests by executing
the following within the outer ``pyndl`` folder:

.. code:: bash

    py.test
    py.test doc/souce/example.rst


For full tests you can run:

.. code:: bash

    tox -e test

For manually checking coding guidelines run:

.. code:: bash

    pycodestyle pyndl tests
    pylint --ignore-patterns='.*\.so' --rcfile=setup.cfg -j 2 pyndl tests

For more details on which tests run in the continuous testing environment
look at the file ``tox.ini``.


Deployment
==========
In order to create a source dist package run:

.. code:: bash

    python3 setup.py sdist


Contributing
============
Please read
`CONTRIBUTING.rst
<https://github.com/quantling/pyndl/blob/master/CONTRIBUTING.rst>`_ for details
on our code of conduct and the process for submitting pull requests to us.


Versioning
==========
At the moment we are still in alpha and therefore no stable API is guaranteed
between minor version increments, but soon we will change to use `SemVer
<http://semver.org/>`_ for versioning. For the versions available, see the
`tags on this repository <https://github.com/quantling/pyndl/tags>`_.


Authors
=======
See also the list of `contributors
<https://github.com/quantling/pyndl/contributors>`_ who participated in this
project.


License
=======
This project is licensed under the MIT License - see the `LICENSE.txt
<https://github.com/quantling/pyndl/blob/master/LICENSE.txt>`_ file for details


Acknowledgments
===============
This package is build as a python replacement for the R ndl2 package. Some
ideas on how to build the API and how to efficiently run the Rescorla Wagner
iterative learning on large text corpora are inspired by the way the ndl2
package solves this problems. The ndl2 package will be published to github in
August 2017 and a reference will be placed here.
