============
Readme pyndl
============

.. image:: https://api.travis-ci.org/quantling/pyndl.svg
    :target: https://travis-ci.com/quantling/pyndl

.. image:: https://coveralls.io/repos/github/quantling/pyndl/badge.svg?branch=master
    :target: https://coveralls.io/github/quantling/pyndl?branch=master

.. image:: https://zenodo.org/badge/80022085.svg
    :target: https://zenodo.org/badge/latestdoi/80022085


This python3 package is a collection of useful script in order to run tasks on
huge amounts of text file corpora. Especially, it allows to efficiently apply
the Rescorla-Wagner learning rule to these corpora.

.. warning::

    This package is still in alpha and there might be some API changes in the
    near future. If you want to suggest us some contact us under konstantin
    (dot) sering (ät) uni-tuebingen.de.

.. note::

    This package is not intended to be used under python2.


Installation
============
Install with::

    python setup.py install [--user]

or development install with::

    python setup.py develop

in order to build a source package run::

    python setup.py sdist


Usage
=====
This package is intended to be used as a python package in (small) python
script or via the ipython REPL. For example code and explanations look into
``doc/source/examples.rst``.


Development
===========


Documentation
-------------
The documentation and the doc-strings within the source code should follow the
numpy doc-string conventions (which are used by pandas as well).

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

http://pandas.pydata.org/pandas-docs/stable/contributing.html#contributing-to-the-documentation


Install
-------
You can install this package with pip from a folder, git repository, or sdist
bundle::

    pip install ~/pyndl/|git+ssh://git@github.com/<user>/pyndl.git|pyndl-<version>.tar.gz


Testing
=======
Check if the package does what it is supposed to do by running the test with
``py.test``::

    py.test-3 --cov-report html --cov=pyndl

The more general way of testing - including unit tests, documentation checks
and style checking - can be done via tox. Tox does testing in a virtual
environment and installs all dependencies. ::

    (pip install tox)
    tox


Important terminology
=====================
Some terminology used in these modules and scripts which descripe specific
behaviour:

cue :
    A cue is something that gives a hint on something else. The something else
    is called outcome. Examples for cues in a text corpus are trigraphs or
    preceeding words for the word or meaning of the word.

outcome :
    A something that happens or is the result of an event. Examples are words,
    the meaning of the word, or lexomes.

event :
    An event connects cues with outcomes. In any event one or more unordered
    cues are present and one or more outcomes are present.

weights :
    The weights represent the learned experience / association between all cues
    and outcomes of interest. Usually, some meta data is stored alongside the
    learned weights.

cue file :
    A cue file contains a list of all cues that are interesting for a specific
    question. It is a utf-8 encoded tab delimitered text file with a header in
    the first line. It has two columns. The first column contains the cue and
    the second column contains the frequency of the cue. There is one cue per
    line. The ordering does not matter.

outcome file :
    An outcome file contains a list of all outcomes that are interesting for a
    specific question. It is a utf-8 encoded tab delimitered text file with a
    header in the first line. It has two columns. The first column contains the
    outcome and the second column contains the frequency of the outcome. There
    is one outcome per line. The ordering does not matter.

symbol file :
    A symbol file contains a list of all symbols that are allowed to occur in
    an outcome or a cue. It is a utf-8 encoded tab delimitered text file with a
    header in the first line. It has two columns. The first column contains the
    symbol and the second column contains the frequency of the symbol. There is
    one symbol per line. The ordering does not matter.

event file :
    An event file contains a list of all events that should be learned. The
    learning will start at the first event and continue to the last event in
    order of the lines. The event file is a utf-8 encoded tab delimitered text
    file with a header in the first line. It has three columns. The first
    column contains an underscore delimitered list of all cues. The second
    column contains an underscore delimitered list of all outcomes. The third
    column contains the frequency of the event. The ordering of the cues and
    outcomes does not matter. There is one event per line. The ordering of the
    lines in the file *does* matter.

corpus file :
    A corpus file is a utf-8 encoded text file that contains huge amounts of
    text. A ``---end.of.document---`` or ``---END.OF.DOCUMENT---`` string marks
    where an old document finished and a new document starts.

weights file :
    A weights file contains the learned weights between cues and outcomes. The
    netCDF format is used to store these information along side with meta data,
    which contains the learning parameters, the time needed to calculate the
    weights, the version of the software used and other information.
