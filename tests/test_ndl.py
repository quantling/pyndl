#!/usr/bin/env python3
# run py.test-3 from the above folder

from collections import defaultdict
import os
import time

import pytest
import numpy as np

from pyndl import ndl, count

slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"),
                          reason="need --runslow option to run")

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
# FILE_PATH = os.path.join(TEST_ROOT, "resources/event_file_tiny.tab")
FILE_PATH_SIMPLE = os.path.join(TEST_ROOT, "resources/event_file_simple.tab")
FILE_PATH_MULTIPLE_CUES = os.path.join(TEST_ROOT, "resources/event_file_multiple_cues.tab")
BINARY_PATH = os.path.join(TEST_ROOT, "binary_resources/")
# REFERENCE_PATH = os.path.join(TEST_ROOT, 'reference/weights_event_file_tiny.csv')
REFERENCE_PATH = os.path.join(TEST_ROOT, 'reference/weights_event_file_simple.csv')
REFERENCE_PATH_NDL2 = os.path.join(TEST_ROOT, 'reference/weights_event_file_simple_ndl2.csv')
REFERENCE_PATH_MULTIPLE_CUES_NDL2 = os.path.join(TEST_ROOT, 'reference/weights_event_file_multiple_cues_ndl2.csv')

LAMBDA_ = 1.0
ALPHA = 0.1
BETAS = (0.1, 0.1)


# Test internal consistency

def test_dict_ndl_vs_thread_ndl_simple():
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS)
    result_thread_ndl_simple = ndl.thread_ndl_simple(FILE_PATH_SIMPLE, ALPHA, BETAS)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_thread_ndl_simple,
                                            is_np_arr1=False, is_np_arr2=True)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_multiple_cues_dict_ndl_vs_thread_ndl_simple():
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, make_unique=True)
    result_thread_ndl_simple = ndl.thread_ndl_simple(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, make_unique=True)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_MULTIPLE_CUES, result_dict_ndl,
                                            result_thread_ndl_simple,
                                            is_np_arr1=False, is_np_arr2=True)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


def test_dict_ndl_vs_openmp_ndl_simple():
    result_dict_ndl = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS)
    result_openmp_ndl_simple = ndl.openmp_ndl_simple(FILE_PATH_SIMPLE, ALPHA, BETAS)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_dict_ndl,
                                            result_openmp_ndl_simple,
                                            is_np_arr1=False, is_np_arr2=True)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


# Test against external ndl2 results
def test_compare_weights_ndl2():
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

    result_python = ndl.dict_ndl(FILE_PATH_SIMPLE, ALPHA, BETAS)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_ndl2, result_python,
                                            is_np_arr1=False, is_np_arr2=False)
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

    result_python = ndl.dict_ndl(FILE_PATH_MULTIPLE_CUES, ALPHA, BETAS, make_unique=False)

    unequal, unequal_ratio = compare_arrays(FILE_PATH_MULTIPLE_CUES, result_ndl2, result_python,
                                            is_np_arr1=False, is_np_arr2=False)
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

    unequal, unequal_ratio = compare_arrays(FILE_PATH_SIMPLE, result_ndl2, result_rescorla,
                                            is_np_arr1=False, is_np_arr2=False)
    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0


@slow
def test_compare_time_dict_inplace_parallel_thread():
    file_path = os.path.join(TEST_ROOT, 'resources/minigeco_wordcues_mini.tab')
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path, number_of_processes=2)

    result_dict_ndl, duration_not_parallel = clock(ndl.dict_ndl, (file_path, ALPHA, BETAS, LAMBDA_))

    result_thread_ndl, duration_parallel = clock(ndl.thread_ndl_simple,
                                                 (file_path, ALPHA, BETAS, LAMBDA_),
                                                 number_of_threads=4)

    assert len(result_dict_ndl) == len(result_thread_ndl)

    unequal, unequal_ratio = compare_arrays(file_path, result_thread_ndl, result_dict_ndl, is_np_arr2=False)
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


def compare_arrays(file_path, arr1, arr2, *, is_np_arr1=True, is_np_arr2=True):
    cues, outcomes = count.cues_outcomes(file_path)
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path, number_of_processes=2)

    cue_indices = [cue_map[cue] for cue in cues]
    outcome_indices = [outcome_map[outcome] for outcome in outcomes]
    unequal = list()

    for outcome in outcomes:
        for cue in cues:
            if is_np_arr1:
                outcome_index = outcome_map[outcome]
                cue_index = cue_map[cue]
                value1 = arr1[outcome_index][cue_index]
            else:
                value1 = arr1[outcome][cue]

            if is_np_arr2:
                outcome_index = outcome_map[outcome]
                cue_index = cue_map[cue]
                value2 = arr2[outcome_index][cue_index]
            else:
                value2 = arr2[outcome][cue]

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
