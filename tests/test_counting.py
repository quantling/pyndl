#!/usr/bin/env python3
# run py.test-3 from the above folder

from .. import counting

def test_cues_outcomes():
    cues, outcomes = counting.cues_outcomes("./tests/events.tab")
    cues3, outcomes3 = counting.cues_outcomes("./tests/events.tab",
                                              number_of_processes=3,
                                              verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = counting.words_symbols("./tests/corpus.txt")
    words3, symbols3 = counting.words_symbols("./tests/corpus.txt",
                                              number_of_processes=3,
                                              verbose=False)
    assert words == words3
    assert symbols == symbols3

