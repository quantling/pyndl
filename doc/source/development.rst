Development
===========
.. image:: https://travis-ci.org/quantling/pyndl.svg?branch=master
    :target: https://travis-ci.org/quantling/pyndl?branch=master

.. image:: https://landscape.io/github/quantling/pyndl/master/landscape.svg?style=flat
    :target: https://landscape.io/github/quantling/pyndl/master
    :alt: Code Health

.. image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. image:: https://img.shields.io/github/issues/quantling/pyndl.svg
    :target: https://github.com/quantling/pyndl/issues

.. image:: https://img.shields.io/github/issues-pr/quantling/pyndl.svg
    :target: https://github.com/quantling/pyndl/pulls


Getting Involved
----------------

The *pyndl* project welcomes help in the following ways:

    * Making Pull Requests for
      `code <https://github.com/quantling/pyndl/tree/master/pyndl>`_,
      `tests <https://github.com/quantling/pyndl/tree/master/tests>`_
      or `documentation <https://github.com/quantling/pyndl/tree/master/doc>`_.
    * Commenting on `open issues <https://github.com/quantling/pyndl/issues>`_
      and `pull requests <https://github.com/quantling/pyndl/pulls>`_.
    * Helping to answer `questions in the issue section
      <https://github.com/quantling/pyndl/labels/question>`_.
    * Creating feature requests or adding bug reports in the `issue section
      <https://github.com/quantling/pyndl/issues/new>`_.


Workflow
--------

1. Fork this repository on Github. From here on we assume you successfully
   forked this repository to https://github.com/yourname/pyndl.git

2. Get a local copy of your fork and install the package in 'development'
   mode, which will make changes in the source code active immediately, by running

.. code:: bash

    git clone https://github.com/yourname/pyndl.git
    cd pyndl
    python3 setup.py develop --user

3. Add code, tests or documentation. You can reference relevant issues in
   commit messages (like #42) to make GitHub link issues and commits together,
   and with phrase like "fixes #42" you can even close relevant issues
   automatically.

4. Push your local changes to your fork:

.. code:: bash

    git push git@github.com:yourname/pyndl.git

5. Open the Pull Requests page at https://github.com/yourname/pyndl/pulls and
   click "New pull request" to submit your Pull Request to
   https://github.com/quantling/pyndl.

.. note::

    If you want to develop *pyndl* you should have installed:

    .. code:: bash

        pip3 install --user tox pylint pytest pycodestyle sphinx


Running tests
-------------

We use ``tox`` to manage testing. You can run the tests by executing the
following within the repository's root folder:

.. code:: bash

    tox -e test

For manually checking coding guidelines run:

.. code:: bash

    tox -e checkstyle
    tox -e lint


Building documentation
----------------------

The projects documentation is stored in the ``doc/`` folder and is created with
``sphinx``. You can rebuild the documentation by either executing

.. code:: bash

   make documentation

in the repository's root folder or by executing

.. code:: bash

   make html

in the documentation folder.


Continous Integration
---------------------

We use several services in order to continuously monitor our project:

===========  ===========  =================  ===========================
Service      Status       Config file        Description
===========  ===========  =================  ===========================
Travis CI    |travis|     `.travis.yml`_     Automated testing
Coveralls    |coveralls|                     Monitoring of test coverage
Landscape    |landscape|  `.landscape.yml`_  Monitoring of code quality
===========  ===========  =================  ===========================

.. |travis| image:: https://travis-ci.org/quantling/pyndl.svg?branch=master
    :target: https://travis-ci.org/quantling/pyndl?branch=master

.. |landscape| image:: https://landscape.io/github/quantling/pyndl/master/landscape.svg?style=flat
    :target: https://landscape.io/github/quantling/pyndl/master

.. |coveralls| image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. _.travis.yml: https://github.com/quantling/pyndl/blob/master/.travis.yml

.. _.landscape.yml: https://github.com/quantling/pyndl/blob/master/.landscape.yml



Release Process
---------------

1. Merge Pull Requests with new features or bugfixes into *pyndl*'s' ``master``
   branch. Ensure, that the version is adequately increased (``X.Y+1.Z`` for new
   features and ``X.Y.Z+1`` for a bugfix).
2. Create a new release on Github of the `master` branch of the form ``vX.Y.Z``.
   Add a description of the new feature or bugfix

3. Pull the repository and checkout the tag and create the distribution files
   using

.. code:: bash

    git pull
    git checkout vX.Y.Z
    python setup.py sdist

4. Create GPG signatures of the distribution files using

.. code:: bash

    gpg --detach-sign -a dist/pyndl-X.Y.Z.tar.gz

5. (maintainers only) Upload the distribution files to PyPI using twine.

.. code:: bash

    twine upload -s dist/*
