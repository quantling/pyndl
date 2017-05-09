===================
Contribute to pyndl
===================

In order to participate in `pyndl` development it is best to fork the
repository on github, then clone your forked repository to your machine and
install it in development mode. Run tests and style checks afterwards.

.. code:: bash

    pip3 uninstall pyndl  # remove old pyndl installation
    git clone git@github.com:<YOUR_USER_NAME>/pyndl.git
    cd pyndl
    python setup.py develop
    tox


Documentation
=============
The documentation and the doc-strings within the source code should follow the
numpy doc-string conventions (which are used by pandas as well).

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

http://pandas.pydata.org/pandas-docs/stable/contributing.html#contributing-to-the-documentation


Building documentation
----------------------
You need to have sphinx (http://www.sphinx-doc.org/en/stable/) installed.

.. code::

    cd doc/
    make html
    make latexpdf

