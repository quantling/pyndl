Development
===========
.. image:: https://travis-ci.com/quantling/pyndl.svg?branch=master
    :target: https://travis-ci.com/quantling/pyndl?branch=master

.. image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. image:: https://img.shields.io/lgtm/grade/python/g/quantling/pyndl.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/quantling/pyndl/context:python

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
       python setup.py develop --user

3. Add code, tests or documentation.

4. Test your changes locally by running within the root folder (``pyndl/``)

   .. code:: bash

       make checkstyle
       make test

5. Add and commit your changes after tests run through without complaints.

   .. code:: bash

       git add -u
       git commit -m 'fixes #42 by posing the question in the right way'

   You can reference relevant issues in commit messages (like #42) to make GitHub
   link issues and commits together, and with phrase like "fixes #42" you can
   even close relevant issues automatically.

6. Push your local changes to your fork:

   .. code:: bash

       git push git@github.com:yourname/pyndl.git

7. Open the Pull Requests page at https://github.com/yourname/pyndl/pulls and
   click "New pull request" to submit your Pull Request to
   https://github.com/quantling/pyndl.

.. note::

    If you want to develop *pyndl* you should have installed:

    .. code:: bash

        pip install --user tox pylint pytest pycodestyle sphinx


Running tests
-------------

We use ``make`` and ``tox`` to manage testing. You can run the tests by
executing the following within the repository's root folder (``pyndl/``):

.. code:: bash

    make test

For manually checking coding guidelines run:

.. code:: bash

    make checkstyle

There is an additional way to invoke ``pylint`` as a linter with tox by running

.. code:: bash

    tox -e lint

The linting gives still a lot of complaints that need some decisions on how to
fix them appropriately.


Local testing with conda
------------------------

Sometimes it might be useful to test if ``pyndl`` works in a clean python
environment. Besides ``tox`` this is possible with ``conda`` as well. The
commands are as follows:

.. code:: bash

    conda create -n testpyndl
    conda activate testpyndl
    conda install python
    python -c 'from pyndl import ndl; print("success")'  # this should fail
    git clone https://github.com/quantling/pyndl.git
    pip install pyndl
    python -c 'from pyndl import ndl; print("success")'  # this should succeed
    conda deactivate
    conda env remove -n testpyndl


Memory profiling
----------------

Sometimes it is useful to monitory the memory footprint of the python process.
This can be achieved by using ``memory_profiler``
(https://pypi.python.org/pypi/memory_profiler).


CPU profiling of C extensions
-----------------------------

In order to profile Cython or C extensions that are invoked from python ``yep``
is a good tool to do that. ``yep`` builds ontop of ``google-perftools``.
(https://pypi.org/project/yep/)


Keeping a fork in sync with master
----------------------------------

.. note::

    If you have questions regarding ``git`` it is mostly a good start to read
    up on it on github help pages, i. e.
    https://help.github.com/articles/working-with-forks/ .

If you fork the ``pyndl`` project on github.com you might want to keep it in
sync with master. In order to do so, you need to setup a remote repository
within a local ``pyndl`` clone of you fork. This remote repository will point
to the original ``pyndl`` repository and is usually called ``upstream``. In
order to do so run with a Terminal within the cloned pyndl folder:

.. code:: bash

    git remote add upstream https://github.com/quantling/pyndl.git

After having set up the ``upstream`` repository you can manually sync your
local repository by running:

.. code:: bash

    git fetch upstream

In order to sync you ``master`` branch run:

.. code:: bash

    git checkout master
    git merge upstream/master

If the merge cannot be fast-forward, you should resolve any issue now and
commit the manually merged files.

After that you should sync you local repository with you github fork by
running:

.. code:: bash

    git push

Some sources with more explanation:

- https://help.github.com/articles/configuring-a-remote-for-a-fork/
- https://help.github.com/articles/syncing-a-fork/


Building documentation
----------------------

Building the documentation requires some extra dependencies. Therefore, run

.. code:: bash

    pip install -e .[docs]

in the project root directory. This command will install all required
dependencies. The projects documentation is stored in the ``pyndl/doc/`` folder
and is created with ``sphinx``. You can rebuild the documentation by either
executing

.. code:: bash

   make documentation

in the repository's root folder (``pyndl``) or by executing

.. code:: bash

   make html

in the documentation folder (``pyndl/doc/``).


Continuous Integration
----------------------

We use several services in order to continuously monitor our project:

===========  ===========  =================  ===========================
Service      Status       Config file        Description
===========  ===========  =================  ===========================
Travis CI    |travis|     `.travis.yml`_     Automated testing
Coveralls    |coveralls|                     Monitoring of test coverage
LGTM         |lgtm|                          Monitoring code quality
===========  ===========  =================  ===========================

.. |travis| image:: https://travis-ci.com/quantling/pyndl.svg?branch=master
    :target: https://travis-ci.com/quantling/pyndl?branch=master

.. |coveralls| image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. |lgtm| image:: https://img.shields.io/lgtm/grade/python/g/quantling/pyndl.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/quantling/pyndl/context:python

.. _.travis.yml: https://github.com/quantling/pyndl/blob/master/.travis.yml


Licensing
---------

All contributions to this project are licensed under the `MIT license
<https://github.com/quantling/pyndl/blob/master/LICENSE.txt>`_. Exceptions are
explicitly marked.
All contributions will be made available under MIT license if no explicit
request for another license is made and agreed on.


Release Process
---------------
1. Ensure, that the version of the branch to be mered, is adequately increased
   see Versioning_ below.

2. Merge Pull Requests with new features or bugfixes into *pyndl*'s' ``master``
   branch.

3. Create a new release on Github of the `master` branch of the form ``vX.Y.Z``
   (where ``X``, ``Y``, and ``Z`` refer to the new version).  Add a description
   of the new feature or bugfix. For details on the version number see
   Versioning_ below.

4. Pull the repository and checkout the tag and create the distribution files
   using:

.. code:: bash

    git pull
    git checkout vX.Y.Z
    python setup.py build  # to compile *.pyx -> *.c
    python setup.py sdist

5. Create GPG signatures of the distribution files using:

.. code:: bash

    gpg --detach-sign -a dist/pyndl-X.Y.Z.tar.gz

6. (maintainers only) Upload the distribution files to PyPI using twine.

.. code:: bash

    twine upload -s dist/*

7. (maintainers only) Check if the new version is on pypi (https://pypi.python.org/pypi/pyndl/).


Versioning
----------
We use a semvers versioning scheme. Assuming the current version is ``X.Y.Z``
than ``X`` refers to the major version, ``Y`` refers to the minor version and
``Z`` refers to a bugfix version.


Bugfix release
^^^^^^^^^^^^^^
For a bugfix only merge, which does not add any new features and does not
break any existing API increase the bugfix version by one (``X.Y.Z ->
X.Y.Z+1``).

Minor release
^^^^^^^^^^^^^
If a merge adds new features or breaks with the existing API a deprecation
warning has to be supplied which should keep the existing API. The minor
version is increased by one (``X.Y.Z -> X.Y+1.Z``). Deprecation warnings should
be kept until the next major version. They should warn the user that the old
API is only usable in this major version and will not be available any more
with the next major ``X+1.0.0`` release onwards. The deprecation warning should
give the exact version number when the API becomes unavailable and the way of
achieving the same behaviour.

Major release
^^^^^^^^^^^^^
If enough changes are accumulated to justify a new major release, create a new
pull request which only contains the following two changes:

- the change of the version number from ``X.Y.Z`` to ``X+1.0.0``
- remove all the API with deprecation warning introduced in the current
  ``X.Y.Z`` release

