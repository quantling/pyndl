#!/usr/bin/env python3

# pylint: disable=C0111

import os
import gzip

import pytest
import pandas as pd

from pyndl import io

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
FILE_PATH_SIMPLE = os.path.join(TEST_ROOT, "resources/event_file_simple.tab.gz")
TMP_SAVE_PATH = os.path.join(TEST_ROOT, "temp/event_file_simple_copy.tab.gz")
TMP_SAVE_PATH_UNCOMPRESSED = os.path.join(TEST_ROOT, "temp/event_file_simple_copy.tab")


def test_uncompressed():
    data_frame = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    ref_events = list(io.events_from_dataframe(data_frame))
    io.events_to_file(ref_events, TMP_SAVE_PATH_UNCOMPRESSED, compression=None)
    events = io.events_from_file(TMP_SAVE_PATH_UNCOMPRESSED, compression=None)

    unequal, unequal_ratio = compare_events(events, ref_events)

    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition

    os.remove(TMP_SAVE_PATH_UNCOMPRESSED)


def test_events_from_dataframe():
    data_frame = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    events = io.events_from_dataframe(data_frame)
    ref_events = io.events_from_file(FILE_PATH_SIMPLE)

    unequal, unequal_ratio = compare_events(events, ref_events)

    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


def test_events_from_list():
    data_frame = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    event_list = [(cues, outcomes) for cues, outcomes in zip(data_frame['cues'],
                                                             data_frame['outcomes'])]

    events = io.events_from_list(event_list)

    ref_events = io.events_from_file(FILE_PATH_SIMPLE)

    unequal, unequal_ratio = compare_events(events, ref_events)

    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0  # pylint: disable=len-as-condition


def test_events_to_file():
    data_frame = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    io.events_to_file(data_frame, TMP_SAVE_PATH)

    with gzip.open(FILE_PATH_SIMPLE, 'rt') as ref_file:
        with gzip.open(TMP_SAVE_PATH, 'rt') as event_file:

            for line, ref_line in zip(event_file, ref_file):
                assert line == ref_line

    os.remove(TMP_SAVE_PATH)


def test_event_generator_to_file():
    events = io.events_from_file(FILE_PATH_SIMPLE)
    io.events_to_file(events, TMP_SAVE_PATH)

    with gzip.open(FILE_PATH_SIMPLE, 'rt') as ref_file:
        with gzip.open(TMP_SAVE_PATH, 'rt') as event_file:

            for line, ref_line in zip(event_file, ref_file):
                assert line == ref_line

    os.remove(TMP_SAVE_PATH)


def compare_events(events1, events2):
    events1, events2 = list(events1), list(events2)

    assert len(events1) == len(events2)

    unequal = list()

    for (cues1, outcomes1), (cues2, outcomes2) in zip(events1, events2):
        if sorted(cues1) != sorted(cues2) or sorted(outcomes1) != sorted(outcomes2):
            unequal.append(((cues1, outcomes1), (cues2, outcomes2)))

    unequal_ratio = len(unequal) / len(events1)
    return (unequal, unequal_ratio)


def test_safe_write_path(tmp_path):
    base_path = tmp_path / "example.txt"
    assert not (tmp_path / "example.txt").exists()

    path = io.safe_write_path(base_path)
    path.write_text('test-file')
    assert (tmp_path / "example.txt").exists()

    path = io.safe_write_path(base_path)
    path.write_text('test-file')
    print(base_path, path)
    assert (tmp_path / "example-1.txt").exists()

    path = io.safe_write_path(base_path)
    path.write_text('test-file')
    assert (tmp_path / "example-2.txt").exists()

    # custom template, which returns a full path instead of a file name
    path = io.safe_write_path(base_path, template='{path}-{counter}')
    path.write_text('test-file')
    assert (tmp_path / "example.txt-1").exists()

    with pytest.raises(ValueError):
        # assert that pathnames are counting to avoid infinity loops
        path = io.safe_write_path(base_path, template='{path}')
