#!/usr/bin/env python3
# run py.test-3 from the above folder

from collections import defaultdict, OrderedDict
import os
import random
import time

import pytest
import numpy as np


slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"),
                          reason="need --runslow option to run")

from .. import ndl, count, preprocess

TEST_ROOT = os.path.dirname(__file__)
FILE_PATH = os.path.join(TEST_ROOT, "resources/event_file_tiny.tab")
BINARY_PATH = os.path.join(TEST_ROOT, "binary_resources/")
REFERENCE_PATH = os.path.join(TEST_ROOT, 'reference/weights_event_file_tiny.csv')

LAMBDA_ = 1.0
ALPHA = 0.1
BETAS = (0.1, 0.1)

def test_dict_ndl():
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(FILE_PATH, number_of_processes=2)
    alphas, betas = generate_alpha_beta(FILE_PATH, cue_map, outcome_map,
                                        fixed_alpha=ALPHA, fixed_beta=BETAS)

    events = ndl.events(FILE_PATH, frequency=True)
    result_dict_ndl = ndl.dict_ndl(events, alphas, betas, all_outcomes)

    unequal, unequal_ratio = compare_with_ndl2(FILE_PATH,
                                               result_dict_ndl,
                                               REFERENCE_PATH)

    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0

def test_dict_ndl_simple():
    result_dict_ndl_simple = ndl.dict_ndl_simple(FILE_PATH,
                                                 ALPHA,
                                                 BETAS,
                                                 LAMBDA_)

    unequal, unequal_ratio = compare_with_ndl2(FILE_PATH,
                                               result_dict_ndl_simple,
                                               REFERENCE_PATH)

    print('%.2f ratio unequal' % unequal_ratio)
#    print(unequal) # TODO remove after debugging
    assert len(unequal) == 0

def test_thread_ndl_simple():
    result_thread_ndl_simple = ndl.thread_ndl_simple(FILE_PATH,
                                                     ALPHA,
                                                     BETAS,
                                                     LAMBDA_)

    unequal, unequal_ratio = compare_with_ndl2(FILE_PATH,
                                               result_thread_ndl_simple,
                                               REFERENCE_PATH)

    print('%.2f ratio unequal' % unequal_ratio)
    assert len(unequal) == 0

def test_openmp_ndl_simple():
    result_openmp_ndl_simple = ndl.openmp_ndl_simple(FILE_PATH,
                                                     ALPHA,
                                                     BETAS,
                                                     LAMBDA_)

    unequal, unequal_ratio = compare_with_ndl2(FILE_PATH,
                                               result_openmp_ndl_simple,
                                               REFERENCE_PATH)

    print('%.2f ratio unequal' % unequal_ratio)
    #print(unequal) # TODO remove after debugging
    assert len(unequal) == 0

@slow
def test_compare_weights_ndl2():
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('event_file_tiny.tab', alpha=0.1, beta=0.1, lambda=1.0)
        wm <- learner$getWeights()
        write.csv(wm, 'weights_event_file_tiny.csv')

    """
    result_ndl2 = defaultdict(lambda: defaultdict(float))

    with open(REFERENCE_PATH, 'rt') as reference_file:
        first_line = reference_file.readline()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_ndl2[outcome][cue] = float(cue_weights[ii])

    result_python = ndl.dict_ndl_simple(FILE_PATH, ALPHA, BETAS, LAMBDA_)

    unequal = compare_arrays(FILE_PATH, result_ndl2, result_python,
                             is_np_arr1=False, is_np_arr2=False)
    assert len(unequal) == 0

@slow
def test_compare_time_dict_inplace_parallel_thread():
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(FILE_PATH, number_of_processes=2)

    result_dict_ndl, duration_not_parallel = clock(ndl.dict_ndl_simple, (FILE_PATH, ALPHA, BETAS, LAMBDA_))

    result_thread_ndl, duration_parallel = clock(ndl.thread_ndl_simple, (FILE_PATH, ALPHA, BETAS, LAMBDA_), number_of_threads=4)

    print('parallel: %.3e  dict: %.3e' % (duration_parallel, duration_not_parallel))
    assert duration_parallel < duration_not_parallel

    assert len(result_dict_ndl) == len(result_inplace_ndl)
    unequal = compare_arrays(file_path, result_inplace_ndl, result_dict_ndl, is_np_arr2=False)
    print('%.2f ratio unequal' % (len(unequal) / (len(outcome_map) * len(cue_map))))
    assert len(unequal) == 0



def test_slice_list():

    l1 = [0,1,2,3,4,5,6,7,8,9]

    res = ndl.slice_list(l1,2)
    assert res == [[0,1],[2,3],[4,5],[6,7],[8,9]]

    res2 = ndl.slice_list(l1,3)
    assert res2 == [[0,1,2],[3,4,5],[6,7,8],[9]]

def generate_alpha_beta(file_path, cue_map, outcome_map, *, fixed_alpha=0.1,
                        fixed_beta=(0.1,0.1), numpy=False):
    betas = fixed_beta

    events = ndl.events(file_path, frequency=True)

    if numpy:
        alphas = np.zeros(len(cue_map), dtype=float)
        for cues, outcomes in events:
            cue_indices = [cue_map[cue] for cue in cues]
            for cue_index in cue_indices:
                alphas[cue_index]= fixed_alpha
    else:
        alphas = defaultdict(float)
        for cues, outcomes in events:
            for cue in cues:
                alphas[cue] = fixed_alpha

    return (alphas, betas)

def clock(f, args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return result, duration

def compare_arrays(file_path, arr1, arr2,* ,is_np_arr1=True, is_np_arr2=True):

    cues, outcomes = count.cues_outcomes(file_path)
    cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path,number_of_processes=2)

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

    return unequal

def compare_with_ndl2(file_path, wm, reference_file_path):
    """
    Compares a weightmatrix with the weightmatrix calculated by ndl2.

    Parameters
    ==========
    file_path : str
        path to the event file
    wm : dict or numpy.ndarray
        weightmatrix learned on file_path by a pyndl function
    reference_file_path : str
        one value for successful prediction (reward) one for punishment

    Returns
    =======
    unequal : list
        list of the unequal cells
    unequal_ratio: float
        ratio of unequal cells in wm in comparison to the ndl2 weightmatrix

    """

    wm_ndl2 = defaultdict(lambda: defaultdict(float))
    cues_ndl = set()

    with open(reference_file_path, 'rt') as reference_file:
        first_line = reference_file.readline().strip()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.split(',')
            cue = cue.strip('"')
            cues_ndl.add(cue) # TODO remove after debugging
            for ii, outcome in enumerate(outcomes):
                wm_ndl2[outcome][cue] = float(cue_weights[ii])

    if type(wm) == np.ndarray:
        cue_map, outcome_map, all_outcomes = ndl.generate_mapping(file_path,
                                                                  number_of_processes=2)
        is_np_array = True
    else:
        is_np_array = False

    cues, outcomes = count.cues_outcomes(file_path)
    unequal = list()

    assert len(wm_ndl2.keys()) == len(outcomes)
    assert len(cues_ndl) == len(cues)
    # write_weights_to_file("weights_pyndl.csv", wm, cues, outcomes)
    # TODO remove after debugging

    for outcome in outcomes:
        for cue in cues:
            if is_np_array:
                outcome_index = outcome_map[outcome]
                cue_index = cue_map[cue]
                value1 = wm[outcome_index][cue_index]
            else:
                value1 = wm[outcome][cue]

            value2 = wm_ndl2[outcome][cue]

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
