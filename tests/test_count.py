#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

from .. import count

TEST_ROOT = os.path.dirname(__file__)

def test_cues_outcomes():
    cues, outcomes = count.cues_outcomes(os.path.join(TEST_ROOT, "resources/events.tab"))
    cues3, outcomes3 = count.cues_outcomes(os.path.join(TEST_ROOT, "resources/events.tab"),
                                           number_of_processes=3,
                                           verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = count.words_symbols(os.path.join(TEST_ROOT, "resources/corpus.txt"))
    words3, symbols3 = count.words_symbols(os.path.join(TEST_ROOT, "resources/corpus.txt"),
                                           number_of_processes=3,
                                           verbose=False)
    assert words == words3
    assert symbols == symbols3


def test_save_load():
    file_name = os.path.join(TEST_ROOT, "temp/cues.tab")
    cues, outcomes = count.cues_outcomes(os.path.join(TEST_ROOT, "resources/event_file.tab"))
    count.save_counter(cues, file_name)
    cues_loaded = count.load_counter(file_name)
    assert cues == cues_loaded
    os.remove(file_name)
