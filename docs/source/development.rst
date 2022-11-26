Development
===========
.. image:: https://github.com/quantling/pyndl/actions/workflows/python-test.yml/badge.svg?branch=main
    :target: https://github.com/quantling/pyndl/actions/workflows/python-test.yml

.. image:: https://codecov.io/gh/quantling/pyndl/branch/main/graph/badge.svg?token=2GWUXRA9PD
    :target: https://codecov.io/gh/quantling/pyndl

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
      `code <https://github.com/quantling/pyndl/tree/main/pyndl>`_,
      `tests <https://github.com/quantling/pyndl/tree/main/tests>`_
      or `documentation <https://github.com/quantling/pyndl/tree/main/doc>`_.
    * Commenting on `open issues <https://github.com/quantling/pyndl/issues>`_
      and `pull requests <https://github.com/quantling/pyndl/pulls>`_.
    * Helping to answer `questions in the issue section
      <https://github.com/quantling/pyndl/labels/question>`_.
    * Creating feature requests or adding bug reports in the `issue section
      <https://github.com/quantling/pyndl/issues/new>`_.


Prerequisites
-------------
To make changes to the *pyndl* code base the following prerequisites need to be
fulfilled on your machine:

* you have Python3 installed
* you have `Cython <https://cython.readthedocs.io>`_ installed and can compile
  Cython extensions (`conda install cython` should do the trick, but sometimes
  this can be a little bit tricky)
* you have `poetry <https://python-poetry.org/>`_ installed
* you have `git` installed

.. note::

   Depending on your operating system and your architecture properly installing
   Cython and being able to compile Cython extensions can be a bit tricky. If
   the installation of python fails, it is a good first step to check that the
   Cython installation is done properly via `conda` or through you package
   manager.


Workflow
--------

1. Fork this repository on Github. From here on we assume you successfully
   forked this repository to https://github.com/yourname/pyndl.git

2. Install all dependencies with poetry (https://python-poetry.org/)

   .. code:: bash

       git clone https://github.com/yourname/pyndl.git
       cd pyndl
       poetry install

3. Add code, tests or documentation.

4. Test your changes locally by running within the root folder (``pyndl/``)

   .. code:: bash

        poetry run pytest
        poetry run pylint pyndl

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


Running tests
-------------

We use ``poetry`` to manage testing. You can run the tests by
executing the following within the repository's root folder (``pyndl/``):

.. code:: bash

    poetry run pytest

Some tests only work on Linux, therefore to run tests on Mac OS or Windows skip
all Linux-only tests with:

.. code:: bash

    poetry run pytest --on-linux

For extensive, time and memory consuming tests run (at least 12 GB of free
memory should be available):

.. code:: bash

    poetry run pytest --run-slow

For manually checking coding guidelines run:

.. code:: bash

    poetry run pylint pyndl

The linting gives still a lot of complaints that need some decisions on how to
fix them appropriately.

.. note::

    Previous versions of *pyndl* used ``make`` and ``tox`` to manage testing. For
    documentation on this, please check the respective version documentations


Local testing with conda
------------------------

Sometimes it might be useful to test if ``pyndl`` works in a clean python
environment. Besides ``poetry`` this is possible with ``conda`` as well. The
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


Keeping a fork in sync with main
----------------------------------

.. note::

    If you have questions regarding ``git`` it is mostly a good start to read
    up on it on github help pages, i. e.
    https://help.github.com/articles/working-with-forks/ .

If you fork the ``pyndl`` project on github.com you might want to keep it in
sync with main. In order to do so, you need to setup a remote repository
within a local ``pyndl`` clone of you fork. This remote repository will point
to the original ``pyndl`` repository and is usually called ``upstream``. In
order to do so run with a Terminal within the cloned pyndl folder:

.. code:: bash

    git remote add upstream https://github.com/quantling/pyndl.git

After having set up the ``upstream`` repository you can manually sync your
local repository by running:

.. code:: bash

    git fetch upstream

In order to sync you ``main`` branch run:

.. code:: bash

    git checkout main
    git merge upstream/main

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

Building the documentation requires some extra dependencies. Usually, these are
installed when installing the dependencies with poetry. Some services like Readthedocs,
however, require the documentation dependencies extra. For that reason, they can
also be found in `docs/requirements.txt`. For normal usage, installing all dependencies
with poetry is sufficient.

The projects documentation is stored in the ``pyndl/docs/`` folder
and is created with ``sphinx``. However, it is not necessary to build the documentation
from there.

You can rebuild the documentation by either executing

.. code:: bash

    poetry run sphinx-build -b html docs/source docs/build/html

in the repository's root folder (``pyndl``) or by executing

.. code:: bash

   poetry run make html

in the documentation folder (``pyndl/docs/``).


Continuous Integration
----------------------

We use several services in order to continuously monitor our project:

===============  ===========  ==================  ===========================
Service          Status       Config file         Description
===============  ===========  ==================  ===========================
Github Actions   |actions|    `python-test.yml`_  Automated testing
Codecov          |codecov|                        Monitoring of test coverage
LGTM             |lgtm|                           Monitoring code quality
===============  ===========  ==================  ===========================

.. |actions| image:: https://github.com/quantling/pyndl/actions/workflows/python-test.yml/badge.svg?branch=main
    :target: https://github.com/quantling/pyndl/actions/workflows/python-test.yml

.. |codecov| image:: https://codecov.io/gh/quantling/pyndl/branch/main/graph/badge.svg?token=2GWUXRA9PD
    :target: https://codecov.io/gh/quantling/pyndl

.. |lgtm| image:: https://img.shields.io/lgtm/grade/python/g/quantling/pyndl.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/quantling/pyndl/context:python

.. _python-test.yml: https://github.com/quantling/pyndl/blob/main/.github/workflows/python-test.yml


Licensing
---------

All contributions to this project are licensed under the `MIT license
<https://github.com/quantling/pyndl/blob/main/LICENSE.txt>`_. Exceptions are
explicitly marked.
All contributions will be made available under MIT license if no explicit
request for another license is made and agreed on.


Release Process
---------------
1. Update the version accordingly to Versioning_ below. This can be easily done
   by poetry running

   .. code:: bash

       poetry version major|minor|patch|...


2. Merge Pull Requests with new features or bugfixes into *pyndl*'s' ``main``
   branch.

3. Create a new release on Github of the `main` branch of the form ``vX.Y.Z``
   (where ``X``, ``Y``, and ``Z`` refer to the new version).  Add a description
   of the new feature or bugfix. For details on the version number see
   Versioning_ below. This will trigger a Action to automatically build and
   upload the release to PyPI

4. Check if the new version is on pypi (https://pypi.python.org/pypi/pyndl/).


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
