#!/usr/bin/env python3

# pylint: disable=C0111

import os

from pyndl import count

TEST_ROOT = os.path.dirname(__file__)
EVENT_RESOURCE_FILE = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word.tab.gz")
CORPUS_RESOURCE_FILE = os.path.join(TEST_ROOT, "resources/corpus.txt")


def test_cues_outcomes():
    n_events, cues, outcomes = count.cues_outcomes(EVENT_RESOURCE_FILE)
    n_events3, cues3, outcomes3 = count.cues_outcomes(EVENT_RESOURCE_FILE,
                                                      n_jobs=6,
                                                      verbose=True)
    assert n_events == 2772
    assert n_events == n_events3
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = count.words_symbols(CORPUS_RESOURCE_FILE)
    words3, symbols3 = count.words_symbols(CORPUS_RESOURCE_FILE,
                                           n_jobs=3,
                                           verbose=True)
    assert words == words3
    assert symbols == symbols3


def test_save_load():
    file_name = os.path.join(TEST_ROOT, "temp/cues.tab")
    _, cues, _ = count.cues_outcomes(EVENT_RESOURCE_FILE)
    count.save_counter(cues, file_name)
    cues_loaded = count.load_counter(file_name)
    assert cues == cues_loaded
    os.remove(file_name)
