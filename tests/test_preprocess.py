#!/usr/bin/env python3
# run py.test-3 from the above folder

import os

import pytest

from ..preprocess import (create_event_file, filter_event_file,
                          create_binary_event_files, bandsample,
                          event_generator, write_events,
                          _job_binary_event_file, JobFilter)

from ..count import cues_outcomes

def test_bandsample():
    cue_freq_map, outcome_freq_map = cues_outcomes("./tests/event_file.tab",
                                                   number_of_processes=2)
    outcome_freq_map_filtered = bandsample(outcome_freq_map, 50, cutoff=1, seed=1234, verbose=True)


def test_create_event_file_bad_symbols():
    with pytest.raises(ValueError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                          "abcd#", context_structure="document",
                          event_structure="consecutive_words", event_option=3)
    with pytest.raises(ValueError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                          "abcd_", context_structure="document",
                          event_structure="consecutive_words", event_option=3)


def test_create_event_file_bad_event_context():
    with pytest.raises(NotImplementedError):
        create_event_file("./tests/corpus.txt", "./tests/events_corpus.tab",
                          context_structure="UNREASONABLE",
                          event_structure="consecutive_words", event_option=3)


def test_create_event_file_upper_case():
    event_file = "./tests/events_corpus_upper_case.tab"
    create_event_file("./tests/corpus.txt", event_file,
                      context_structure="document",
                      event_structure="consecutive_words", event_option=3)
    os.remove(event_file)


def test_create_event_file_trigrams_to_word():
    event_file = "./tests/event_file_trigrams_to_word.tab"
    create_event_file("./tests/corpus_tiny.txt", event_file,
                      context_structure="document",
                      event_structure="consecutive_words", event_option=3,
                      cue_structure="trigrams_to_word")
    with open(event_file, "rt") as new_file:
        lines_new = new_file.readlines()
    with open("./tests/event_file_trigrams_to_word_reference.tab", "rt") as reference:
        lines_reference = reference.readlines()
    assert lines_new == lines_reference
    os.remove(event_file)


def test_create_event_file_bigrams_to_word():
    event_file = "./tests/event_file_bigrams_to_word.tab"
    create_event_file("./tests/corpus_tiny.txt", event_file,
                      context_structure="document",
                      event_structure="consecutive_words", event_option=3,
                      cue_structure="bigrams_to_word")
    with open(event_file, "rt") as new_file:
        lines_new = new_file.readlines()
    with open("./tests/event_file_bigrams_to_word_reference.tab", "rt") as reference:
        lines_reference = reference.readlines()
    assert lines_new == lines_reference
    os.remove(event_file)



def test_create_event_file_word_to_word():
    event_file = "./tests/event_file_word_to_word.tab"
    create_event_file("./tests/corpus_tiny.txt", event_file,
                      context_structure="document",
                      event_structure="consecutive_words", event_option=3,
                      cue_structure="word_to_word")
    with open(event_file, "rt") as new_file:
        lines_new = new_file.readlines()
    with open("./tests/event_file_word_to_word_reference.tab", "rt") as reference:
        lines_reference = reference.readlines()
    assert lines_new == lines_reference
    os.remove(event_file)


def test_filter_event_file_bad_event_file():
    input_event_file = "./tests/event_file_BAD.tab"
    output_event_file = "./tests/event_file_BAD_output.tab"
    with pytest.raises(ValueError):
        filter_event_file(input_event_file, output_event_file)
    os.remove(output_event_file)


def test_job_filter():
    allowed_cues = ["#of", "of#"]
    allowed_outcomes = ["of",]
    job = JobFilter(allowed_cues, allowed_outcomes)
    line = '#of_alb_NEI_b_of#_XX\tterm_not_of\t3\n'
    new_line = job.job(line)
    assert(new_line == '#of_of#\tof\t3\n')
    # no cues
    line = 'alb_NEI_b_XX\tterm_not_of\t3\n'
    new_line = job.job(line)
    assert(new_line is None)
    # no outcomes
    line = '#of_alb_NEI_b_of#_XX\tterm_not\t3\n'
    new_line = job.job(line)
    assert(new_line is None)
    # neither cues nor outcomes
    line = '#alb_NEI_b_XX\tterm_not\t3\n'
    new_line = job.job(line)
    assert(new_line is None)
    with pytest.raises(ValueError):
        bad_line = 'This is a bad line.'
        job.job(bad_line)


def test_filter_event_file():
    input_event_file = "./tests/event_file.tab"
    output_event_file = "./tests/event_file_filtered.tab"
    cues = ["#of", "of#"]
    cues.sort()
    outcomes = ["of",]
    outcomes.sort()
    filter_event_file(input_event_file, output_event_file,
                      allowed_cues=cues,
                      allowed_outcomes=outcomes,
                      number_of_processes=2,
                      verbose=True)
    cue_freq_map, outcome_freq_map = cues_outcomes(output_event_file)
    cues_new = list(cue_freq_map)
    cues_new.sort()
    outcomes_new = list(outcome_freq_map)
    outcomes_new.sort()
    assert cues == cues_new
    assert outcomes == outcomes_new
    os.remove(output_event_file)


def test_write_events():
    event_file = "./tests/event_file.tab"
    cue_freq_map, outcome_freq_map = cues_outcomes(event_file)
    outcomes = list(outcome_freq_map.keys())
    outcomes.sort()
    cues = list(cue_freq_map.keys())
    cues.sort()
    cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    file_name = "./tests/events.bin"
    with pytest.raises(StopIteration):
        write_events(events, file_name)
    os.remove(file_name)

    # start stop
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    write_events(events, file_name, start=10, stop=20)
    os.remove(file_name)

    # no events
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    write_events(events, file_name, start=100000, stop=100010)

    _job_binary_event_file(file_name=file_name, event_file=event_file,
                           cue_id_map=cue_id_map,
                           outcome_id_map=outcome_id_map,
                           sort_within_event=False,
                           start=0, stop=10)
    os.remove(file_name)

    # bad event file
    with pytest.raises(ValueError):
        events = event_generator("./tests/event_file_BAD.tab", cue_id_map,
                                outcome_id_map)
        # traverse generator
        for event in events:
            pass


def test_preprocessing():
    corpus_file = "./tests/corpus.txt"
    event_file = "./tests/events_corpus.tab"
    symbols = "abcdefghijklmnopqrstuvwxyzóąćęłńśźż"  # polish

    # create event file
    create_event_file(corpus_file, event_file, symbols,
                      context_structure="document",
                      event_structure="consecutive_words", event_option=3,
                      lower_case=True, verbose=True)

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
    for file_ in os.listdir(path_name):
        os.remove(os.path.join(path_name, file_))
    os.rmdir(path_name)
    create_binary_event_files(path_name, event_file_filtered, cue_id_map,
                              outcome_id_map, sort_within_event=False,
                              number_of_processes=2, events_per_file=1000,
                              verbose=True)
    with pytest.raises(IOError):
        create_binary_event_files(path_name, event_file_filtered, cue_id_map,
                                outcome_id_map, sort_within_event=False,
                                number_of_processes=2, events_per_file=1000,
                                verbose=True)
    create_binary_event_files(path_name, event_file_filtered, cue_id_map,
                            outcome_id_map, sort_within_event=False,
                            number_of_processes=2, events_per_file=1000,
                            overwrite=True, verbose=True)


    # clean everything
    os.remove(event_file)
    os.remove(event_file_filtered)

