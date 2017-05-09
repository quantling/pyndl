#!/usr/bin/env python3

# pylint: disable=C0111

from collections import defaultdict, OrderedDict
import os
import time
import tempfile
import copy


import pytest
import numpy as np
import xarray as xr
import pandas as pd

from pyndl import ndl, count

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"),
                          reason="need --runslow option to run")

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
FILE_PATH_SIMPLE = os.path.join(TEST_ROOT, "resources/event_file_simple.tab.gz")
FILE_PATH_MULTIPLE_CUES = os.path.join(TEST_ROOT, "resources/event_file_multiple_cues.tab.gz")
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
def result_dict_ndl_generator():
    return ndl.dict_ndl(ndl.events_from_file(FILE_PATH_SIMPLE), ALPHA, BETAS)


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

    part_path_1 = os.path.join(TMP_PATH, "event_file_simple_1.tab.gz")
    part_path_2 = os.path.join(TMP_PATH, "event_file_simple_2.tab.gz")

    part_1.to_csv(part_path_1, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"],
                  compression='gzip')
    part_2.to_csv(part_path_2, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"],
                  compression='gzip')

    del events_simple, part_1, part_2

    result_part = ndl.ndl(part_path_1,
                          ALPHA, BETAS)

    result = ndl.ndl(part_path_2, ALPHA, BETAS,
                     weights=result_part)

    return result


def test_exceptions():
    with pytest.raises(ValueError) as e_info:
        ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='threading', weights=1)
        assert e_info == 'weights need to be None or xarray.DataArray with method=threading'

    with pytest.raises(ValueError) as e_info:
        ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='magic')
        assert e_info == 'method needs to be either "threading" or "openmp"'

    with pytest.raises(ValueError) as e_info:
        ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, weights=1)
        assert e_info == 'weights needs to be either defaultdict or None'

    with pytest.raises(ValueError) as e_info:
        ndl.dict_ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, remove_duplicates=None)
        assert e_info == 'cues or outcomes needs to be unique: cues "a a"; outcomes "A"; use remove_duplicates=True'

    with pytest.raises(ValueError) as e_info:
        ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='threading',
                len_sublists=-1)
        assert e_info == "'len_sublists' must be larger then one"

    with pytest.raises(ValueError) as e_info:
        ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, make_data_array="magic")
        assert e_info == "make_data_array must be True or False"

    with pytest.raises(ValueError) as e_info:
        ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, remove_duplicates="magic")
        assert e_info == "remove_duplicates must be None, True or False"

    with pytest.raises(ValueError) as e_info:
        ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, remove_duplicates="magic")
        assert e_info == "remove_duplicates must be None, True or False"


def test_continue_learning_dict():
    events_simple = pd.read_csv(FILE_PATH_SIMPLE, sep="\t")
    part_1 = events_simple.head(CONTINUE_SPLIT_POINT)
    part_2 = events_simple.tail(len(events_simple) - CONTINUE_SPLIT_POINT)

    assert len(part_1) > 0
    assert len(part_2) > 0

    part_path_1 = os.path.join(TMP_PATH, "event_file_simple_1.tab.gz")
    part_path_2 = os.path.join(TMP_PATH, "event_file_simple_2.tab.gz")

    part_1.to_csv(part_path_1, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"],
                  compression='gzip')
    part_2.to_csv(part_path_2, header=True, index=None,
                  sep='\t', columns=["cues", "outcomes"],
                  compression='gzip')

    del events_simple, part_1, part_2

    result_part = ndl.dict_ndl(part_path_1,
                               ALPHA, BETAS)
    result_part_copy = copy.deepcopy(result_part)

    result_inplace = ndl.dict_ndl(part_path_2, ALPHA, BETAS,
                                  weights=result_part, inplace=True)

    assert result_part is result_inplace
    assert result_part != result_part_copy

    result_part = ndl.dict_ndl(part_path_1,
                               ALPHA, BETAS)

    result = ndl.dict_ndl(part_path_2,
                          ALPHA, BETAS, weights=result_part)

    assert result_part != result


def test_continue_learning_dict_ndl_data_array(result_dict_ndl, result_dict_ndl_data_array):
    continue_from_dict = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS,
                                      weights=result_dict_ndl)
    continue_from_data_array = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS,
                                            weights=result_dict_ndl_data_array)
    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE,
                                            continue_from_dict,
                                            continue_from_data_array)
    print(continue_from_data_array)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_continue_learning(result_continue_learning, result_ndl_openmp):
    assert result_continue_learning.shape == result_ndl_openmp.shape

    assert set(result_continue_learning.coords["outcomes"].values) == set(result_ndl_openmp.coords["outcomes"].values)

    assert set(result_continue_learning.coords["cues"].values) == set(result_ndl_openmp.coords["cues"].values)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE,
                                            result_continue_learning,
                                            result_ndl_openmp)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_save_to_netcdf4(result_ndl_openmp):
    weights = result_ndl_openmp.copy()  # avoids changing shared test data
    path = os.path.join(TMP_PATH, "weights.nc")
    weights.to_netcdf(path)
    weights_read = xr.open_dataarray(path)
    # does not preserves the order of the OrderedDict
    for key, value in weights.attrs.items():
        assert value == weights_read.attrs[key]
    weights_continued = ndl.ndl(FILE_PATH_SIMPLE, ALPHA, BETAS, method='openmp', weights=weights)
    path_continued = os.path.join(TMP_PATH, "weights_continued.nc")
    weights_continued.to_netcdf(path_continued)
    weights_continued_read = xr.open_dataarray(path_continued)
    for key, value in weights_continued.attrs.items():
        assert value == weights_continued_read.attrs[key]


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


def test_dict_ndl_vs_dict_ndl_generator(result_dict_ndl, result_dict_ndl_generator):
    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_dict_ndl_generator)
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


def test_meta_data(result_dict_ndl, result_dict_ndl_data_array, result_ndl_openmp, result_ndl_threading):
    attributes = {'cython', 'cpu_time', 'hostname', 'xarray', 'wall_time',
                  'event_path', 'number_events', 'username', 'method', 'date', 'numpy',
                  'betas', 'lambda', 'pyndl', 'alpha', 'pandas', 'method',
                  'function'}
    results = [result_dict_ndl, result_dict_ndl_data_array, result_ndl_threading, result_ndl_openmp]
    for result in results:
        assert set(result.attrs.keys()) == attributes

    assert int(result_dict_ndl_data_array.attrs['number_events']) > 0
    assert len(set(
        [result.attrs['number_events'].strip()
         for result in results]
    )) == 1


# Test against external ndl2 results
def test_compare_weights_ndl2(result_dict_ndl):
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('event_file_simple.tab.gz', alpha=0.1, beta=0.1, lambda=1.0)
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
        learner <- learnWeightsTabular('tests/resources/event_file_multiple_cues.tab.gz',
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
        learner <- learnWeightsTabular('tests/resources/event_file_simple.tab.gz', alpha=0.1, beta=0.1, lambda=1.0)
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
    file_path = os.path.join(TEST_ROOT, 'resources/event_file_many_cues.tab.gz')
    cue_map, outcome_map, all_outcomes = generate_mapping(file_path)

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
    n_events, cues, outcomes = count.cues_outcomes(file_path)
    cue_map, outcome_map, all_outcomes = generate_mapping(file_path)

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
                    values.append(array.loc[{'outcomes': outcome, 'cues': cue}].values)
                elif isinstance(array, pd.DataFrame):
                    values.append(array.loc[outcome][cue])
                else:
                    values.append(array[outcome][cue])

            value1, value2 = values
            if not np.isclose(value1, value2, rtol=1e-02, atol=1e-05):
                unequal.append((outcome, cue, value1, value2))

    unequal_ratio = len(unequal) / (len(outcomes) * len(cues))
    return (unequal, unequal_ratio)


def generate_mapping(event_path):
    n_events, cues, outcomes = count.cues_outcomes(event_path)
    all_cues = list(cues.keys())
    all_outcomes = list(outcomes.keys())
    cue_map = OrderedDict(((cue, ii) for ii, cue in enumerate(all_cues)))
    outcome_map = OrderedDict(((outcome, ii) for ii, outcome in enumerate(all_outcomes)))

    return (cue_map, outcome_map, all_outcomes)
