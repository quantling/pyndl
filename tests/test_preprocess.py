#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

import pytest

from ..preprocess import (create_event_file, filter_event_file,
                          create_binary_event_files, bandsample)

from ..count import cues_outcomes


def test_create_event_file_bad_symbols():
    with pytest.raises(ValueError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                         "abcd#", context="document",
                         event="consecutive_words", event_option=3)
    with pytest.raises(ValueError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                         "abcd_", context="document",
                         event="consecutive_words", event_option=3)

def test_create_event_file_bad_event_context():
    with pytest.raises(NotImplementedError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                         context="UNREASONABLE", event="consecutive_words",
                         event_option=3)

def test_create_event_file_upper_case():
    event_file = "./tests/events_corpus_upper_case.tab"
    create_event_file("./tests/corpus.txt", event_file,
                        context="document", event="consecutive_words",
                        event_option=3)
    os.remove(event_file)


def test_filter_event_file_bad_event_file():
    input_event_file = "./tests/event_file_BAD.tab"
    output_event_file = "./tests/event_file_BAD_output.tab"
    with pytest.raises(ValueError):
        filter_event_file(input_event_file, output_event_file)
    os.remove(output_event_file)


def test_filter_event_file():
    input_event_file = "./tests/event_file.tab"
    output_event_file = "./tests/event_file_filtered.tab"
    cues = ["#of", "of#"]
    cues.sort()
    outcomes = ["of",]
    outcomes.sort()
    filter_event_file(input_event_file, output_event_file,
                     allowed_cues=cues,
                     allowed_outcomes=outcomes)
    cue_freq_map, outcome_freq_map = cues_outcomes(output_event_file)
    cues_new = list(cue_freq_map)
    cues_new.sort()
    outcomes_new = list(outcome_freq_map)
    outcomes_new.sort()
    assert cues == cues_new
    assert outcomes == outcomes_new
    os.remove(output_event_file)


def test_preprocessing():
    corpus_file = "./tests/corpus.txt"
    event_file = "./tests/events_corpus.tab"
    symbols = "abcdefghijklmnopqrstuvwxyzóąćęłńśźż"  # polish

    # create event file
    create_event_file(corpus_file, event_file, symbols,
                     context="document", event="consecutive_words",
                     event_option=3, lower_case=True, verbose=True)

    # read in cues and outcomes
    cue_freq_map, outcome_freq_map = cues_outcomes(event_file,
                                                   number_of_processes=2)
    cues = list(cue_freq_map.keys())
    cues.sort()
    cue_id_map = {cue: ii for ii, cue in enumerate(cues)}

    # reduce number of outcomes through bandsampling
    outcome_freq_map_filtered = bandsample(outcome_freq_map, 50, cutoff=1, seed=1234)
    outcomes = list(outcome_freq_map_filtered.keys())
    outcomes.sort()
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}

    # filter outcomes by reduced number of outcomes
    event_file_filtered = event_file + ".filtered"
    filter_event_file(event_file, event_file_filtered, allowed_outcomes=outcomes)


    # create binary event files
    path_name = event_file_filtered + ".events"
    create_binary_event_files(path_name, event_file_filtered, cue_id_map,
                              outcome_id_map, sort_within_event=False,
                              number_of_processes=2, events_per_file=1000, overwrite=True,
                              verbose=True)

    # clean everything
    os.remove(event_file)
    os.remove(event_file_filtered)

