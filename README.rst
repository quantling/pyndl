===============================================
Pyndl - Naive Discriminative Learning in Python
===============================================

.. image:: https://github.com/quantling/pyndl/actions/workflows/python-test.yml/badge.svg?branch=main
    :target: https://github.com/quantling/pyndl/actions/workflows/python-test.yml

.. image:: https://codecov.io/gh/quantling/pyndl/branch/main/graph/badge.svg?token=2GWUXRA9PD
    :target: https://codecov.io/gh/quantling/pyndl

.. image:: https://img.shields.io/pypi/pyversions/pyndl.svg
    :target: https://pypi.python.org/pypi/pyndl/

.. image:: https://img.shields.io/github/license/quantling/pyndl.svg
    :target: https://github.com/quantling/pyndl/blob/main/LICENSE

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.597964.svg
   :target: https://doi.org/10.5281/zenodo.597964

*pyndl* is an implementation of Naive Discriminative Learning in Python. It was
created to analyse huge amounts of text file corpora. Especially, it allows to
efficiently apply the Rescorla-Wagner learning rule to these corpora.


Installation
============

The easiest way to install *pyndl* is using
`pip <https://pip.pypa.io/en/stable/>`_:

.. code:: bash

    pip install --user pyndl

For more information have a look at the `Installation Guide
<http://pyndl.readthedocs.io/en/latest/installation.html>`_.


Documentation
=============

*pyndl* uses ``sphinx`` to create a documentation manual. The documentation is
hosted on `Read the Docs <http://pyndl.readthedocs.io/en/latest/>`_.


Getting involved
================

The *pyndl* project welcomes help in the following ways:

* Making Pull Requests for
  `code <https://github.com/quantling/pyndl/tree/main/pyndl>`_,
  `tests <https://github.com/quantling/pyndl/tree/main/tests>`_
  or `documentation <https://github.com/quantling/pyndl/tree/main/doc>`_.
* Commenting on `open issues <https://github.com/quantling/pyndl/issues>`_
  and `pull requests <https://github.com/quantling/pyndl/pulls>`_.
* Helping to answer `questions in the issue section
  <https://github.com/quantling/pyndl/labels/question>`_.
* Creating feature requests or adding bug reports in the `issue section
  <https://github.com/quantling/pyndl/issues/new>`_.

For more information on how to contribute to *pyndl* have a look at the
`development section <http://pyndl.readthedocs.io/en/latest/development.html>`_.


Authors and Contributers
========================

*pyndl* was mainly developed by
`Konstantin Sering <https://github.com/derNarr>`_,
`Marc Weitz <https://github.com/trybnetic>`_,
`David-Elias Künstle <https://github.com/dekuenstle/>`_,
`Elnaz Shafaei Bajestan <https://github.com/elnazsh>`_
and `Lennart Schneider <https://github.com/sumny>`_. For the full list of
contributers have a look at `Github's Contributor summary
<https://github.com/quantling/pyndl/contributors>`_.

Currently, it is maintained by `Konstantin Sering <https://github.com/derNarr>`_
and `Marc Weitz <https://github.com/trybnetic>`_.


Funding
-------
*pyndl* was partially funded by the Humboldt grant, the ERC advanced grant (no.
742545) and by the University of Tübingen.


Acknowledgements
----------------
This package is build as a python replacement for the R ndl2 package. Some
ideas on how to build the API and how to efficiently run the Rescorla Wagner
iterative learning on large text corpora are inspired by the way the ndl2
package solves this problems. The ndl2 package is available on Github `here
<https://github.com/quantling/ndl2>`_.

