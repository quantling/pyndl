#!/usr/bin/env python3
# run py.test-3 from the above folder

from .. import count

def test_cues_outcomes():
    cues, outcomes = count.cues_outcomes("./tests/events.tab")
    cues3, outcomes3 = count.cues_outcomes("./tests/events.tab",
                                              number_of_processes=3,
                                              verbose=False)
    assert cues == cues3
    assert outcomes == outcomes3


def test_words_symbols():
    words, symbols = count.words_symbols("./tests/corpus.txt")
    words3, symbols3 = count.words_symbols("./tests/corpus.txt",
                                              number_of_processes=3,
                                              verbose=False)
    assert words == words3
    assert symbols == symbols3

