#!/usr/bin/env python3
# run py.test-3 from the above folder

from collections import defaultdict
import os
import time
import tempfile


import pytest
import numpy as np
import xarray as xr
import pandas as pd

from pyndl import ndl, count

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"),
                          reason="need --runslow option to run")

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
FILE_PATH_SIMPLE = os.path.join(TEST_ROOT, "resources/event_file_simple.tab")
FILE_PATH_MULTIPLE_CUES = os.path.join(TEST_ROOT, "resources/event_file_multiple_cues.tab")
REFERENCE_PATH = os.path.join(TEST_ROOT, 'reference/weights_event_file_simple.csv')
REFERENCE_PATH_NDL2 = os.path.join(TEST_ROOT, 'reference/weights_event_file_simple_ndl2.csv')
REFERENCE_PATH_MULTIPLE_CUES_NDL2 = os.path.join(TEST_ROOT, 'reference/weights_event_file_multiple_cues_ndl2.csv')

TMP_PATH = tempfile.mkdtemp()

LAMBDA_ = 1.0
ALPHA = 0.1
BETAS = (0.1, 0.1)

CONTINUE_SPLIT_POINT = 3


@pytest.fixture(scope='module')
def result_ndl_threading():
    return ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='threading')


@pytest.fixture(scope='module')
def result_ndl_openmp():
    return ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='openmp')


@pytest.fixture(scope='module')
def result_dict_ndl():
    return ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS)


@pytest.fixture(scope='module')
def result_dict_ndl_data_array():
    return ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, make_data_array=True)


@pytest.fixture(scope='module')
def result_continue_learning():
    events_simple = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    part_1 = events_simple.head(CONTINUE_SPLIT_POINT)
    part_2 = events_simple.tail(len(events_simple) - CONTINUE_SPLIT_POINT)

    assert len(part_1) > 0
    assert len(part_2) > 0

    part_path_1 = os.path.join(TMP_PATH, "event_file_simple_1.tab")
    part_path_2 = os.path.join(TMP_PATH, "event_file_simple_2.tab")

    part_1.to_csv(part_path_1, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"])
    part_2.to_csv(part_path_2, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"])

    del events_simple, part_1, part_2

    result_part = ndl.ndl(part_path_1,
                          ALPHA, BETAS)

    result_part = ndl.ndl(part_path_2,
                          ALPHA, BETAS, weights=result_part)

    return result_part


def test_continue_learning(result_continue_learning, result_ndl_openmp):
    assert result_continue_learning.shape == result_ndl_openmp.shape

    assert set(result_continue_learning.coords["outcomes"].values) == set(result_ndl_openmp.coords["outcomes"].values)

    assert set(result_continue_learning.coords["cues"].values) == set(result_ndl_openmp.coords["cues"].values)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE,
                                            result_continue_learning,
                                            result_ndl_openmp)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_return_values(result_dict_ndl, result_dict_ndl_data_array, result_ndl_threading, result_ndl_openmp):
    # dict_ndl
    assert isinstance(result_dict_ndl, defaultdict)
    assert isinstance(result_dict_ndl_data_array, xr.DataArray)
    # openmp
    assert isinstance(result_ndl_openmp, xr.DataArray)
    # threading
    assert isinstance(result_ndl_threading, xr.DataArray)


# Test internal consistency

def test_dict_ndl_vs_ndl_threading(result_dict_ndl, result_ndl_threading):
    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_ndl_threading)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_dict_ndl_data_array_vs_ndl_threading(result_ndl_threading):
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, make_data_array=True)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_ndl_threading)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_multiple_cues_dict_ndl_vs_ndl_threading():
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, remove_duplicates=True)
    result_ndl_threading = ndl.ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, remove_duplicates=True, method='threading')

    unequal, unequal_ratio = compare_arrays(FILE_PATH_MULTIPLE_CUES, result_dict_ndl,
                                            result_ndl_threading)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_dict_ndl_vs_ndl_openmp(result_dict_ndl, result_ndl_openmp):
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS)
    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_ndl_openmp)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_meta_data(result_dict_ndl_data_array, result_ndl_openmp, result_ndl_threading):
    attributes = {'cython', 'cpu_time', 'hostname', 'xarray', 'wall_time',
                  'event_path', 'username', 'time', 'method', 'date', 'numpy',
                  'betas', 'lambda', 'pyndl', 'alpha', 'pandas'}
    assert set(result_ndl_openmp.attrs.keys()) == attributes
    assert set(result_ndl_threading.attrs.keys()) == attributes
    #assert set(result_dict_ndl_data_array.attrs.keys()) == attributes


# Test against external ndl2 results
def test_compare_weights_ndl2(result_dict_ndl):
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('event_file_simple.tab', alpha=0.1, beta=0.1, lambda=1.0)
        wm <- learner$getWeights()
        wm <- wm[order(rownames(wm)), order(colnames(wm))]
        write.csv(wm, 'weights_event_file_simple_ndl2.csv')

    """
    result_ndl2 = defaultdict(lambda: defaultdict(float))

    with open(REFERENCE_PATH, 'rt') as reference_file:
        first_line = reference_file.readline().strip()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.strip().split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_ndl2[outcome][cue] = float(cue_weights[ii])

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_ndl2, result_dict_ndl)
    print(set(outcome for outcome, *_ in unequal))
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_multiple_cues_dict_ndl_vs_ndl2():
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('tests/resources/event_file_multiple_cues.tab',
                                       alpha=0.1, beta=0.1, lambda=1.0, removeDuplicates=FALSE)
        wm <- learner$getWeights()
        wm <- wm[order(rownames(wm)), order(colnames(wm))]
        write.csv(wm, 'tests/reference/weights_event_file_multiple_cues_ndl2.csv')

    """
    result_ndl2 = defaultdict(lambda: defaultdict(float))

    with open(REFERENCE_PATH_MULTIPLE_CUES_NDL2, 'rt') as reference_file:
        first_line = reference_file.readline().strip()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.strip().split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_ndl2[outcome][cue] = float(cue_weights[ii])

    result_python = ndl.dict_ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, remove_duplicates=False)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_MULTIPLE_CUES, result_ndl2, result_python)
    print(set(outcome for outcome, *_ in unequal))
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_compare_weights_rescorla_vs_ndl2():
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('tests/resources/event_file_simple.tab', alpha=0.1, beta=0.1, lambda=1.0)
        wm <- learner$getWeights()
        wm <- wm[order(rownames(wm)), order(colnames(wm))]
        write.csv(wm, 'tests/reference/weights_event_file_simple_ndl2.csv')

    """
    result_ndl2 = defaultdict(lambda: defaultdict(float))

    with open(REFERENCE_PATH, 'rt') as reference_file:
        first_line = reference_file.readline().strip()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.strip().split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_ndl2[outcome][cue] = float(cue_weights[ii])

    result_rescorla = defaultdict(lambda: defaultdict(float))

    with open(REFERENCE_PATH_NDL2, 'rt') as reference_file:
        first_line = reference_file.readline().strip()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.strip().split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_rescorla[outcome][cue] = float(cue_weights[ii])

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_ndl2, result_rescorla)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


@slow
def test_compare_time_dict_inplace_parallel_thread():
    file_path = os.path.join(TEST_ROOT, 'resources/event_file_many_cues.tab')
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path, number_of_processes=2)

    result_dict_ndl, duration_not_parallel = clock(ndl.dict_ndl, (file_path, ALPHA, BETAS, LAMBDA_))

    result_thread_ndl, duration_parallel = clock(ndl.ndl,
                                                 (file_path, ALPHA, BETAS, LAMBDA_),
                                                 number_of_threads=4, method='threading')

    assert len(result_dict_ndl) == len(result_thread_ndl)

    unequal, unequal_ratio = compare_arrays(file_path, result_thread_ndl, result_dict_ndl)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0

    print('parallel: %.3e  dict: %.3e' % (duration_parallel, duration_not_parallel))
    assert duration_parallel < duration_not_parallel


def test_slice_list():
    l1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    res = ndl.slice_list(l1, 2)
    assert res == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    res2 = ndl.slice_list(l1, 3)
    assert res2 == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def clock(f, args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return result, duration


def compare_arrays(file_path, arr1, arr2):
    cues, outcomes = count.cues_outcomes(file_path)
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path, number_of_processes=2)

    cue_indices = [cue_map[cue] for cue in cues]
    outcome_indices = [outcome_map[outcome] for outcome in outcomes]
    unequal = list()

    for outcome in outcomes:
        for cue in cues:
            values = list()
            for array in (arr1, arr2):
                if isinstance(array, np.ndarray):
                    outcome_index = outcome_map[outcome]
                    cue_index = cue_map[cue]
                    values.append(array[outcome_index][cue_index])
                elif isinstance(array, xr.DataArray):
                    values.append(array.loc[{'outcomes': outcome, 'cues': cue}])
                elif isinstance(array, pd.DataFrame):
                    values.append(array.loc[outcome][cue])
                else:
                    values.append(array[outcome][cue])

            value1, value2 = values
            if not np.isclose(value1, value2, rtol=1e-02, atol=1e-05):
                unequal.append((outcome, cue, value1, value2))

    unequal_ratio = len(unequal) / (len(outcomes) * len(cues))
    return (unequal, unequal_ratio)


def write_weights_to_file(file_path, weights, cues, outcomes):
    if type(weights) == np.ndarray:
        is_np_array = True
    else:
        is_np_array = False
    with open(file_path, 'w') as o_file:
        o_file.write('""')
        for outcome in sorted(outcomes):
            o_file.write(',"%s"' % outcome)
        o_file.write("\n")
        for cue in sorted(cues):
            o_file.write('"%s"' % cue)
            for outcome in sorted(outcomes):
                if is_np_array:
                    outcome_index = outcome_map[outcome]
                    cue_index = cue_map[cue]
                    value = weights[outcome_index][cue_index]
                else:
                    value = weights[outcome][cue]
                o_file.write(',%s' % value)
            o_file.write("\n")
