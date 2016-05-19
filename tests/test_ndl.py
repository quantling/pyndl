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


from .. import ndl, count

TEST_ROOT = os.path.dirname(__file__)

file_path = os.path.join(TEST_ROOT, 'resources/event_file_tiny.tab')#minigeco_wordcues_mini.tab')
reference_path = os.path.join(TEST_ROOT, 'reference/weights_event_tiny_R_ndl2.csv')
cues, outcomes = count.cues_outcomes(file_path)
all_outcomes = list(outcomes.keys())

cue_map, outcome_map = ndl.generate_mapping(file_path,number_of_processes=2)

alpha = 0.1
betas = (0.1,0.1)


def test_compare_weights_numpy():
    """
    Checks whether the output of the numpy and the dict
    implementation of ndl is equal.

    """

    result_dict_ndl = ndl.dict_ndl_simple(ndl.events(file_path, frequency=True), alpha, betas, all_outcomes)
    result_numpy_ndl = ndl.numpy_ndl_simple(file_path, alpha, betas, all_outcomes,frequency=True)

    assert len(result_numpy_ndl) == len(result_dict_ndl)
    #assert len(result_numpy_ndl[0]) == len(result_dict_ndl[0])

    unequal = list()
    for outcome, cues in result_dict_ndl.items():
        for cue in cues:
            if not np.isclose(result_dict_ndl[outcome][cue], result_numpy_ndl[outcome_map[outcome]][cue_map[cue]], rtol=0, atol=0):
                unequal.append((outcome, cue, result_dict_ndl[outcome][cue], result_numpy_ndl[outcome_map[outcome]][cue_map[cue]]))

    #print(unequal)
    print('%.2f ratio unequal' % (len(unequal) / (len(result_numpy_ndl) * len(list(result_numpy_ndl[0])))))
    assert len(unequal) == 0


def test_compare_weights_parallel():
    """
    Checks whether the output of the parallel and the not parallel
    implementation of dict_ndl is equal.

    """

    alphas, betas = generate_random_alpha_beta(file_path)

    events = ndl.events(file_path, frequency=True)
    result_not_parallel = ndl.dict_ndl(events, alphas, betas, all_outcomes)
    result_parallel = ndl.dict_ndl_parrallel(file_path, alphas, betas, all_outcomes, frequency_in_event_file=True)


    for outcome, cue_dict in result_parallel.items():
        for cue in cue_dict:
            assert result_parallel[outcome][cue] == result_not_parallel[outcome][cue]


@slow
def test_compare_weights_ndl2():
    """
    Checks whether the output of the R learner implemented in ndl2 and the
    python implementation of dict_ndl is equal.

    R code to generate the results::

        library(ndl2)
        learner <- learnWeightsTabular('resources/bnc_tri2l_1k.events', alpha=0.1, beta=0.1)
        wm <- learner$getWeights()
        write.csv(wm, 'reference/weights_bnc_tri2l_1k_R_ndl2.csv')

    """
    result_ndl2 = defaultdict(lambda: defaultdict(float))

    with open(reference_path, 'rt') as reference_file:
        first_line = reference_file.readline()
        outcomes = first_line.split(',')[1:]
        outcomes = [outcome.strip('"') for outcome in outcomes]
        for line in reference_file:
            cue, *cue_weights = line.split(',')
            cue = cue.strip('"')
            for ii, outcome in enumerate(outcomes):
                result_ndl2[outcome][cue] = float(cue_weights[ii])

    file_path = os.path.join(TEST_ROOT, 'resources/bnc_tri2l_1k.events')
    alphas = betas = defaultdict(lambda: 0.1)
    events = ndl.events(file_path, frequency=True)
    result_python = ndl.dict_ndl(events, alphas, betas, all_outcomes)

    unequal = list()
    for outcome, cue_dict in result_python.items():
        for cue in cue_dict:
            if not np.isclose(result_ndl2[outcome][cue], result_python[outcome][cue], rtol=1e-02, atol=1e-05):
                unequal.append((outcome, cue, result_ndl2[outcome][cue], result_python[outcome][cue]))

    print(unequal)
    print('%.2f ratio unequal' % (len(unequal) / (len(result_python.keys()) * len(list(result_python.values())[0].keys()))))
    assert len(unequal) == 0



@slow
def test_compare_time_parallel():
    """
    Compares the times to execute the implementations of dict_ndl.

    """

    # we need a bigger event file for the timing
    cues, outcomes = count.cues_outcomes(file_path)
    all_outcomes = list(outcomes.keys())

    alphas, betas = generate_random_alpha_beta(file_path)

    result_not_parallel, duration_not_parrallel = clock(ndl.dict_ndl, (file_path, alphas, betas, all_outcomes))

    result_parallel, duration_parrallel = clock(ndl.dict_ndl_parrallel, (file_path, alphas, betas, all_outcomes))

    # For small files this test is expected to fail. Otherwise it is expected
    # that a parrallel implementation of dict_ndl should be faster.
    assert duration_parrallel < duration_not_parrallel
    for outcome, cue_dict in result_parallel.items():
        for cue in cue_dict:
            assert result_parallel[outcome][cue] == result_not_parallel[outcome][cue]


def test_slice_list():

    l1 = [0,1,2,3,4,5,6,7,8,9]

    res = ndl.slice_list(l1,2)
    assert res == [[0,1],[2,3],[4,5],[6,7],[8,9]]

    res2 = ndl.slice_list(l1,3)
    assert res2 == [[0,1,2],[3,4,5],[6,7,8],[9]]



def generate_random_alpha_beta(file_path):
    alphas = defaultdict(float)
    betas = (0.3, 0.1)

    events = ndl.events(file_path, frequency=True)

    for cues, outcomes in events:
        for cue in cues:
            alphas[cue] = 0.3 #random.random()

    return (alphas, betas)


def clock(f, args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return result, duration
