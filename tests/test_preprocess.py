#!/usr/bin/env python3

# pylint: disable=C0111

from collections import OrderedDict
import gzip
import os

import pytest

from pyndl.preprocess import (create_event_file, filter_event_file,
                              create_binary_event_files, bandsample,
                              event_generator, write_events,
                              _job_binary_event_file, JobFilter, to_bytes, to_integer, read_binary_file)

from pyndl.count import (cues_outcomes, load_counter, save_counter)
from pyndl import ndl

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
EVENT_FILE = os.path.join(TEST_ROOT, "temp/events_corpus.tab.gz")
RESOURCE_FILE = os.path.join(TEST_ROOT, "resources/corpus.txt")


def test_bandsample():
    resource_file = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word.tab.gz")
    n_events, cue_freq_map, outcome_freq_map = cues_outcomes(resource_file,
                                                             number_of_processes=2)
    outcome_freq_map_filtered = bandsample(outcome_freq_map, 50, cutoff=1, seed=None, verbose=False)
    assert len(outcome_freq_map_filtered) == 50

    reference_file = os.path.join(TEST_ROOT, 'reference/bandsampled_outcomes.tab')
    try:
        outcome_freq_map_filtered_reference = load_counter(reference_file)
    except (FileNotFoundError):
        temp_file = os.path.join(TEST_ROOT, 'temp/bandsampled_outcomes.tab')
        save_counter(outcome_freq_map_filtered, temp_file)
        raise

    bandsample(outcome_freq_map, 50, cutoff=1, verbose=True)


def test_create_event_file_bad_symbols():
    with pytest.raises(ValueError):
        create_event_file(RESOURCE_FILE, EVENT_FILE,
                          "abcd#")
    assert not os.path.isfile(EVENT_FILE)
    with pytest.raises(ValueError):
        create_event_file(RESOURCE_FILE, EVENT_FILE,
                          "abcd_")
    assert not os.path.isfile(EVENT_FILE)


def test_create_event_file_bad_event_context():
    with pytest.raises(NotImplementedError):
        create_event_file(RESOURCE_FILE, EVENT_FILE,
                          context_structure="UNREASONABLE")
    assert not os.path.isfile(EVENT_FILE)


def test_create_event_file_bad_event_event():
    with pytest.raises(NotImplementedError):
        create_event_file(RESOURCE_FILE, EVENT_FILE,
                          event_structure="UNREASONABLE")
    assert not os.path.isfile(EVENT_FILE)


def test_create_event_file_upper_case():
    event_file = os.path.join(TEST_ROOT, "temp/events_corpus_upper_case.tab.gz")
    create_event_file(RESOURCE_FILE, event_file,
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3, ))
    os.remove(event_file)


def test_create_event_file_trigrams_to_word():
    event_file = os.path.join(TEST_ROOT, "temp/event_file_trigrams_to_word.tab.gz")
    reference_file = os.path.join(TEST_ROOT, "reference/event_file_trigrams_to_word.tab.gz")
    create_event_file(RESOURCE_FILE, event_file,
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3, ),
                      cue_structure="trigrams_to_word")
    compare_event_files(event_file, reference_file)
    os.remove(event_file)


def test_create_event_file_trigrams_to_word_line_based():
    event_file = os.path.join(TEST_ROOT, "temp/event_file_trigrams_to_word_line_based.tab.gz")
    reference_file = os.path.join(TEST_ROOT, "reference/event_file_trigrams_to_word_line_based.tab.gz")
    create_event_file(RESOURCE_FILE, event_file,
                      context_structure="document",
                      event_structure="line", event_options=(3, ),
                      cue_structure="trigrams_to_word")
    compare_event_files(event_file, reference_file)
    os.remove(event_file)


def test_create_event_file_bigrams_to_word():
    event_file = os.path.join(TEST_ROOT, "temp/event_file_bigrams_to_word.tab.gz")
    reference_file = os.path.join(TEST_ROOT, "reference/event_file_bigrams_to_word.tab.gz")
    create_event_file(RESOURCE_FILE, event_file,
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3, ),
                      cue_structure="bigrams_to_word",
                      remove_duplicates=True)
    compare_event_files(event_file, reference_file)
    os.remove(event_file)


def test_create_event_file_word_to_word():
    event_file = os.path.join(TEST_ROOT, "temp/event_file_word_to_word.tab.gz")
    reference_file = os.path.join(TEST_ROOT, "reference/event_file_word_to_word.tab.gz")
    create_event_file(RESOURCE_FILE, event_file,
                      context_structure="document",
                      event_structure="word_to_word", event_options=(2, 1),
                      cue_structure="word_to_word")
    compare_event_files(event_file, reference_file)

    os.remove(event_file)


def test_filter_event_file_bad_event_file():
    input_event_file = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word_BAD.tab.gz")
    output_event_file = os.path.join(TEST_ROOT, "temp/event_file_BAD_output.tab.gz")
    with pytest.raises(ValueError):
        filter_event_file(input_event_file, output_event_file)
    os.remove(output_event_file)


def test_job_filter():
    keep_cues = ["#of", "of#"]
    keep_outcomes = ["of", ]
    job = JobFilter(keep_cues, keep_outcomes, None, None, None, None)
    line = '#of_alb_NEI_b_of#_XX\tterm_not_of\n'
    new_line = job.job(line)
    assert(new_line == '#of_of#\tof\n')
    # no cues
    line = 'alb_NEI_b_XX\tterm_not_of\n'
    new_line = job.job(line)
    assert(new_line is None)
    # no outcomes
    line = '#of_alb_NEI_b_of#_XX\tterm_not\n'
    new_line = job.job(line)
    assert(new_line == '#of_of#\t\n')
    # neither cues nor outcomes
    line = '#alb_NEI_b_XX\tterm_not\n'
    new_line = job.job(line)
    assert(new_line is None)
    with pytest.raises(ValueError):
        bad_line = 'This is a bad line.'
        job.job(bad_line)


def test_filter_event_file():
    input_event_file = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word.tab.gz")
    output_event_file = os.path.join(TEST_ROOT, "temp/event_file_filtered.tab.gz")
    cues = ["#of", "of#"]
    cues.sort()
    outcomes = ["of", ]
    outcomes.sort()
    filter_event_file(input_event_file, output_event_file,
                      keep_cues=cues,
                      keep_outcomes=outcomes,
                      number_of_processes=2,
                      verbose=True)
    n_events, cue_freq_map, outcome_freq_map = cues_outcomes(output_event_file)
    cues_new = list(cue_freq_map)
    cues_new.sort()
    outcomes_new = list(outcome_freq_map)
    outcomes_new.sort()
    assert cues == cues_new
    assert outcomes == outcomes_new
    os.remove(output_event_file)


def test_write_events():
    event_file = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word.tab.gz")
    n_events, cue_freq_map, outcome_freq_map = cues_outcomes(event_file)
    outcomes = list(outcome_freq_map.keys())
    outcomes.sort()
    cues = list(cue_freq_map.keys())
    cues.sort()
    cue_id_map = {cue: ii for ii, cue in enumerate(cues)}
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    file_name = os.path.join(TEST_ROOT, "temp/events.bin")
    with pytest.raises(StopIteration):
        write_events(events, file_name, remove_duplicates=True)
    os.remove(file_name)

    # start stop
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    n_events = write_events(events, file_name, start=10, stop=20, remove_duplicates=True)
    assert n_events == 10
    os.remove(file_name)

    # no events
    events = event_generator(event_file, cue_id_map, outcome_id_map, sort_within_event=True)
    n_events = write_events(events, file_name, start=100000, stop=100010, remove_duplicates=True)
    assert n_events == 0

    _job_binary_event_file(file_name=file_name, event_file=event_file,
                           cue_id_map=cue_id_map,
                           outcome_id_map=outcome_id_map,
                           sort_within_event=False,
                           start=0, stop=10, remove_duplicates=True)
    _job_binary_event_file(file_name=file_name, event_file=event_file,
                           cue_id_map=cue_id_map,
                           outcome_id_map=outcome_id_map,
                           sort_within_event=False,
                           start=0, stop=10, remove_duplicates=True)
    os.remove(file_name)

    # bad event file
    with pytest.raises(ValueError):
        event_bad_file = os.path.join(TEST_ROOT, "resources/event_file_trigrams_to_word_BAD.tab.gz")
        events = event_generator(event_bad_file, cue_id_map,
                                 outcome_id_map)
        # traverse generator
        for event in events:
            pass


def test_byte_conversion():
    a = 184729172
    assert a == to_integer(to_bytes(a))


def test_read_binary_file():
    file_path = "resources/event_file_trigrams_to_word.tab.gz"
    binary_path = "binary_resources/"

    abs_file_path = os.path.join(TEST_ROOT, file_path)
    abs_binary_path = os.path.join(TEST_ROOT, binary_path)
    abs_binary_file_path = os.path.join(abs_binary_path, "events_0_0.dat")

    n_events, cues, outcomes = cues_outcomes(abs_file_path)
    cue_id_map = OrderedDict(((cue, ii) for ii, cue in enumerate(cues.keys())))
    outcome_id_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(outcomes.keys())))

    number_events = create_binary_event_files(abs_file_path, abs_binary_path, cue_id_map,
                                              outcome_id_map, overwrite=True, remove_duplicates=False)

    bin_events = read_binary_file(abs_binary_file_path)
    events = ndl.events_from_file(abs_file_path)
    events_dup = ndl.events_from_file(abs_file_path)

    assert number_events == len(list(events_dup))

    for event, bin_event in zip(events, bin_events):
        cues, outcomes = event
        bin_cues, bin_outcomes = bin_event
        if len(cues) != len(bin_cues):
            raise ValueError('Cues have different length')
        if len(outcomes) != len(bin_outcomes):
            raise ValueError('Cues have different length')

        for cue, bin_cue in zip(cues, bin_cues):
            assert cue_id_map[cue] == bin_cue

        for outcome, bin_outcome in zip(outcomes, bin_outcomes):
            assert outcome_id_map[outcome] == bin_outcome

    # clean everything
    os.remove(abs_binary_file_path)


def test_preprocessing():
    corpus_file = os.path.join(TEST_ROOT, "resources/corpus.txt")
    event_file = os.path.join(TEST_ROOT, "temp/events_corpus.tab.gz")
    symbols = "abcdefghijklmnopqrstuvwxyzóąćęłńśźż"  # polish

    # create event file
    create_event_file(corpus_file, event_file, symbols,
                      context_structure="document",
                      event_structure="consecutive_words",
                      event_options=(3, ),
                      lower_case=True, verbose=True)

    # read in cues and outcomes
    n_events, cue_freq_map, outcome_freq_map = cues_outcomes(event_file,
                                                             number_of_processes=2)
    cues = list(cue_freq_map.keys())
    cues.sort()
    cue_id_map = {cue: ii for ii, cue in enumerate(cues)}

    # reduce number of outcomes through bandsampling
    outcome_freq_map_filtered = bandsample(outcome_freq_map, 50, cutoff=1, seed=None)
    outcomes = list(outcome_freq_map_filtered.keys())
    outcomes.sort()
    outcome_id_map = {outcome: nn for nn, outcome in enumerate(outcomes)}

    # filter outcomes by reduced number of outcomes
    event_file_filtered = event_file + ".filtered"
    filter_event_file(event_file, event_file_filtered, keep_outcomes=outcomes)

    # TODO this is not working at the moment
    # create binary event files
    # path_name = event_file_filtered + ".events"
    # create_binary_event_files(event_file_filtered, path_name, cue_id_map,
    #                           outcome_id_map, sort_within_event=False,
    #                           number_of_processes=2, events_per_file=1000,
    #                           verbose=True)
    # with pytest.raises(IOError):
    #    create_binary_event_files(event_file_filtered, path_name, cue_id_map,
    #                            outcome_id_map, sort_within_event=False,
    #                            number_of_processes=2, events_per_file=1000,
    #                            verbose=True)
    # overwrite=True
    # create_binary_event_files(event_file_filtered, path_name, cue_id_map,
    #                        outcome_id_map, sort_within_event=False,
    #                        number_of_processes=2, events_per_file=1000,
    #                        overwrite=True, verbose=True)

    # clean everything
    os.remove(event_file)
    os.remove(event_file_filtered)
    # for file_ in os.listdir(path_name):
    #    os.remove(os.path.join(path_name, file_))
    # os.rmdir(path_name)


def compare_event_files(newfile, oldfile):
    with gzip.open(newfile, "rt") as new_file:
        lines_new = new_file.readlines()
    with gzip.open(oldfile, "rt") as reference:
        lines_reference = reference.readlines()
    assert len(lines_new) == len(lines_reference)
    for ii in range(len(lines_new)):
        cues, outcomes = lines_new[ii].strip().split('\t')
        cues = sorted(cues.split('_'))
        outcomes = sorted(outcomes.split('_'))
        ref_cues, ref_outcomes = lines_reference[ii].strip().split('\t')
        ref_cues = sorted(ref_cues.split('_'))
        ref_outcomes = sorted(ref_outcomes.split('_'))
        assert cues == ref_cues
        assert outcomes == ref_outcomes
