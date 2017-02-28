#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

from pyndl import count

TEST_ROOT = os.path.dirname(__file__)
EVENT_RESOURCE_FILE = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word.tab")
CORPUS_RESOURCE_FILE = os.path.join(TEST_ROOT, "resources/corpus.txt")

def test_cues_outcomes():
    cues, outcomes = count.cues_outcomes(EVENT_RESOURCE_FILE)
    cues3, outcomes3 = count.cues_outcomes(EVENT_RESOURCE_FILE,
                                           number_of_processes=3,
                                           verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_cues_outcomes_multiprocess():
    cues, outcomes = count.cues_outcomes(EVENT_RESOURCE_FILE)
    cues3, outcomes3 = count.cues_outcomes(EVENT_RESOURCE_FILE,
                                           number_of_processes=6,
                                           verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = count.words_symbols(CORPUS_RESOURCE_FILE)
    words3, symbols3 = count.words_symbols(CORPUS_RESOURCE_FILE,
                                           number_of_processes=3,
                                           verbose=False)
    assert words == words3
    assert symbols == symbols3


def test_save_load():
    file_name = os.path.join(TEST_ROOT, "temp/cues.tab")
    cues, outcomes = count.cues_outcomes(EVENT_RESOURCE_FILE)
    count.save_counter(cues, file_name)
    cues_loaded = count.load_counter(file_name)
    assert cues == cues_loaded
    os.remove(file_name)
