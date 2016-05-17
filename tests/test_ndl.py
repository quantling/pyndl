#!/usr/bin/env python3
# run py.test-3 from the above folder

from collections import defaultdict
import os
import random
import time

from .. import ndl, count

TEST_ROOT = os.path.dirname(__file__)

file_path = os.path.join(TEST_ROOT, 'resources/minigeco_wordcues_mini.tab')
cues, outcomes = count.cues_outcomes(file_path)
all_outcomes = list(outcomes.keys())


def test_compare_output():
    """
    Checks whether the output of the parrallel and the not parrallel implementation of dict_ndl is equal

    """

    alphas, betas = generate_random_alpha_beta(file_path)

    result_not_parallel = ndl.dict_ndl(file_path, alphas, betas, all_outcomes)
    result_parallel = ndl.dict_ndl_parrallel(file_path, alphas, betas, all_outcomes)

    for outcome, cue_dict in result_parallel.items():
        for cue in cue_dict:
            assert result_parallel[outcome][cue] == result_not_parallel[outcome][cue]

def test_compare_time():
    """
    Compares the times to execute the implementations of dict_ndl

    """

    alphas, betas = generate_random_alpha_beta(file_path)

    duration_not_parrallel = clock(ndl.dict_ndl, (file_path, alphas, betas, all_outcomes))

    duration_parrallel = clock(ndl.dict_ndl_parrallel, (file_path, alphas, betas, all_outcomes))

    # For small files this test is expected to fail. Otherwise it is expected
    # that a parrallel implementation of dict_ndl should be faster.
    assert duration_parrallel < duration_not_parrallel

def test_slice_list():

    l1 = [0,1,2,3,4,5,6,7,8,9]

    res = ndl.slice_list(l1,2)
    assert res == [[0,1],[2,3],[4,5],[6,7],[8,9]]

    res2 = ndl.slice_list(l1,3)
    assert res2 == [[0,1,2],[3,4,5],[6,7,8],[9]]




def generate_random_alpha_beta(file_path):
    alphas = defaultdict(float)
    betas = defaultdict(float)

    events = ndl.events(file_path)

    for cues, outcomes in events:
        for cue in cues:
            alphas[cue] = 0.3 #random.random()
        for outcome in outcomes:
            betas[outcome] = 0.3 #random.random()

    return (alphas,betas)

def clock(f, args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return duration
