#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

from .. import count


def test_cues_outcomes():
    cues, outcomes = count.cues_outcomes("./tests/resources/events.tab")
    cues3, outcomes3 = count.cues_outcomes("./tests/resources/events.tab",
                                              number_of_processes=3,
                                              verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = count.words_symbols("./tests/resources/corpus.txt")
    words3, symbols3 = count.words_symbols("./tests/resources/corpus.txt",
                                              number_of_processes=3,
                                              verbose=False)
    assert words == words3
    assert symbols == symbols3


def test_save_load():
    file_name = "./tests/temp/cues.tab"
    cues, outcomes = count.cues_outcomes("./tests/resources/event_file.tab")
    count.save_counter(cues, file_name)
    cues_loaded = count.load_counter(file_name)
    assert cues == cues_loaded
    os.remove(file_name)

