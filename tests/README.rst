==================
README pyndl tests
==================

Running tests
=============
Running tests is described in the documentation in Chapter Development in
Section Running tests. This section can be found here::

   https://pyndl.readthedocs.io/en/latest/development.html#running-tests

The standard set of tests can be execute with::

   poetry run pytest

For this to work `poetry` and `pytest` need to be installed.


Folder 'resources'
==================
The following resources are used by the tests. These are the inputs to the
tests.

corpus.txt
----------
The corpus file is based on the print version of the wikipedia article about
the Rescorla–Wagner-modell (2017-02-28,
https://hu.wikipedia.org/wiki/Rescorla–Wagner-modell, CC BY-SA 3.0) in
Hungarian.  HTML tags were removed mostly automatically with `pandoc
--from=html --to=plain corpus.html`, in addition by hand.  The remaining words
were separated at whitespace to lines by `tr -s '[[:space:]]' '\n'`.

event_file_trigrams_to_word.tab
-------------------------------
Automatically created from corpus.txt according to tests (see
test_preprocess.test_create_event_file_trigrams_to_word).

event_file_trigrams_to_word_BAD.tab
-----------------------------------
Copy of event_file_trigrams_to_word.tab were one line is corrupted because cues
and outcomes are not tab separated.

event_file_simple.tab
---------------------
Short list of simple artificial tab separated events.

event_file_multiple_cues.tab
---------------------
Short list of artificial tab separated events with multiple cues per outcome.

event_file_many_cues.tab
---------------------
1000 events from a real world application with a lot of cues per event.

xml_gz_corpus
-------------
Folder hierarchy containing gzip files with artificial xml subtitles.


Folder 'reference'
==================
The reference folder contains reference data that is used to compare and
validate test results.

event_file_*.tab
----------------
Automatically created from corpus.txt according to tests.

bandsampled_outcomes.tab
------------------------
Outcome frequencies. Automatically generated from
resource/event_file_trigrams_to_word.tab with preprocess.bandsample.

weights_event_file_{simple,multiple_cues}{,_ndl2}.csv
-----------------------------------------------------
Automatically generated weights from event_file_{simple,multiple_cues}.tab

xml_gz_corpus.txt
-----------------
Subtitle text in phrases per line from resource/xml_gz_corpus.


Folder 'temp'
=============
This folder contains temporary files that are written during the testing.

