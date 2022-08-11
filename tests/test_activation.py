#!/usr/bin/env python3

# pylint: disable=C0111

import time
import gc
import os
from collections import defaultdict

import numpy as np
import xarray as xr

import pytest

from pyndl import ndl
from pyndl.activation import activation


TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
FILE_PATH_SIMPLE = os.path.join(TEST_ROOT, "resources/event_file_simple.tab.gz")
FILE_PATH_MULTIPLE_CUES = os.path.join(TEST_ROOT, "resources/event_file_multiple_cues.tab.gz")

LAMBDA_ = 1.0
ALPHA = 0.1
BETAS = (0.1, 0.1)


def test_exceptions():
    with pytest.raises(ValueError) as e_info:
        weights = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, remove_duplicates=None)
        activation(FILE_PATH_MULTIPLE_CUES, weights)
        assert e_info == 'cues or outcomes needs to be unique: cues "a a"; outcomes "A"; use remove_duplicates=True'

    with pytest.raises(ValueError) as e_info:
        activation(FILE_PATH_MULTIPLE_CUES, "magic")
        assert e_info == "Weights other than xarray.DataArray or dicts are not supported."


@pytest.mark.nolinux
def test_activation_matrix():
    weights = xr.DataArray(np.array([[0, 1, 0], [1, 0, 0]]),
                           coords={
                               'outcomes': ['o1', 'o2'],
                               'cues': ['c1', 'c2', 'c3']
                           },
                           dims=('outcomes', 'cues'))

    events = [(['c1', 'c2', 'c3'], []),
              (['c1', 'c3'], []),
              (['c2'], []),
              (['c1', 'c1'], [])]
    reference_activations = np.array([[1, 0, 1, 0], [1, 1, 0, 1]])

    with pytest.raises(ValueError):
        activations = activation(events, weights, n_jobs=1)

    activations = activation(events, weights, n_jobs=1, remove_duplicates=True)
    activations_mp = activation(events, weights, n_jobs=3, remove_duplicates=True)

    assert np.allclose(reference_activations, activations)
    assert np.allclose(reference_activations, activations_mp)


@pytest.mark.nolinux
def test_ignore_missing_cues():
    weights = xr.DataArray(np.array([[0, 1, 0], [1, 0, 0]]),
                           coords={
                               'outcomes': ['o1', 'o2'],
                               'cues': ['c1', 'c2', 'c3']
                           },
                           dims=('outcomes', 'cues'))

    events = [(['c1', 'c2', 'c3'], []),
              (['c1', 'c3'], []),
              (['c2', 'c4'], []),
              (['c1', 'c1'], [])]
    reference_activations = np.array([[1, 0, 1, 0], [1, 1, 0, 1]])

    with pytest.raises(KeyError):
        activations = activation(events, weights, n_jobs=1,
                                 remove_duplicates=True)

    activations = activation(events, weights, n_jobs=1,
                             remove_duplicates=True, ignore_missing_cues=True)
    activations_mp = activation(events, weights, n_jobs=3,
                                remove_duplicates=True, ignore_missing_cues=True)

    assert np.allclose(reference_activations, activations)
    assert np.allclose(reference_activations, activations_mp)


def test_activation_dict():
    weights = defaultdict(lambda: defaultdict(float))
    weights['o1']['c1'] = 0
    weights['o1']['c2'] = 1
    weights['o1']['c3'] = 0
    weights['o2']['c1'] = 1
    weights['o2']['c2'] = 0
    weights['o2']['c3'] = 0
    events = [(['c1', 'c2', 'c3'], []),
              (['c1', 'c3'], []),
              (['c2'], []),
              (['c1', 'c1'], [])]
    reference_activations = {
        'o1': [1, 0, 1, 0],
        'o2': [1, 1, 0, 1]
    }

    with pytest.raises(ValueError):
        activations = activation(events, weights, n_jobs=1)

    activations = activation(events, weights, n_jobs=1, remove_duplicates=True)
    for outcome, activation_list in activations.items():
        assert np.allclose(reference_activations[outcome], activation_list)


def test_ignore_missing_cues_dict():
    weights = defaultdict(lambda: defaultdict(float))
    weights['o1']['c1'] = 0
    weights['o1']['c2'] = 1
    weights['o1']['c3'] = 0
    weights['o2']['c1'] = 1
    weights['o2']['c2'] = 0
    weights['o2']['c3'] = 0
    events = [(['c1', 'c2', 'c3'], []),
              (['c1', 'c3'], []),
              (['c2', 'c4'], []),
              (['c1', 'c1'], [])]
    reference_activations = {
        'o1': [1, 0, 1, 0],
        'o2': [1, 1, 0, 1]
    }

    with pytest.raises(ValueError):
        activations = activation(events, weights, n_jobs=1)

    activations = activation(events, weights, n_jobs=1,
                             remove_duplicates=True, ignore_missing_cues=True)
    for outcome, activation_list in activations.items():
        assert np.allclose(reference_activations[outcome], activation_list)


@pytest.mark.runslow
def test_activation_matrix_large():
    """
    Test with a lot of data. Better run only with at least 12GB free RAM.
    To get time prints for single and multiprocessing run with pytest ...
    --capture=no --runslow

    """
    print("")
    print("Start setup...")

    def time_test(func, of=""):  # pylint: disable=invalid-name
        def dec_func(*args, **kwargs):
            print(f"start test '{of}'")
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            print(f"finished test '{of}'")
            print(f"  duration: {end - start:.3f}s")
            print("")
            return res
        return dec_func

    nn = 2000
    n_cues = 10 * nn
    n_outcomes = nn
    n_events = 10 * nn
    n_cues_per_event = 30
    weight_mat = np.random.rand(n_outcomes, n_cues)
    cues = [f'c{ii}' for ii in range(n_cues)]
    weights = xr.DataArray(weight_mat,
                           coords={'cues': cues},
                           dims=('outcomes', 'cues'))
    events = [(np.random.choice(cues, n_cues_per_event), [])
              for _ in range(n_events)]  # no generator, we use it twice

    print("Start test...")
    print("")
    gc.collect()
    asp = (time_test(activation, of="single threaded")
           (events, weights, n_jobs=1, remove_duplicates=True))
    gc.collect()
    amp = (time_test(activation, of="multi threaded (up to 8 threads)")
           (events, weights, n_jobs=8, remove_duplicates=True))
    del weights
    del events
    gc.collect()
    print("Compare results...")
    assert np.allclose(asp, amp), "single and multi threaded had different results"
    print("Equal.")
