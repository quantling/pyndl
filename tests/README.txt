This file describes the used test resources.

resource folder
================

corpus.txt
----------
The corpus file is based on the print version of the wikipedia article about the Rescorla–Wagner-modell (2017-02-28, https://hu.wikipedia.org/wiki/Rescorla–Wagner-modell, CC BY-SA 3.0) in Hungarian.
HTML tags were removed mostly automatically with `pandoc --from=html --to=plain corpus.html`, in addition by hand.
The remaining words were separated at whitespace to lines by `tr -s '[[:space:]]' '\n'`.

event_file_trigrams_to_word.tab
-------------------------------
Automatically created from corpus.txt according to tests (see test_preprocess.test_create_event_file_trigrams_to_word).

event_file_trigrams_to_word_BAD.tab
-----------------------------------
Copy of event_file_trigrams_to_word.tab were one line is corrupted because cues and outcomes are not tab separated.

reference folder
================

event_file_*.tab
----------------
Automatically created from corpus.txt according to tests.

bandsampled_outcomes.tab
------------------------
Outcome frequencies. Automatically generated from resource/event_file_trigrams_to_word.tab with preprocess.bandsample.
