#!/usr/bin/env python3

# pylint: disable=C0111

import numpy as np

from pyndl import kalman


def DISABLEtest_covariance():
    covmat = np.matrix(np.loadtxt("./tests/covmat.txt"))
    covmat_matrix = covmat.copy()
    covmat_loop = covmat.copy()

    cues_present_vec = np.matrix(np.zeros((covmat.shape[0], 1)), dtype=np.double)
    cues_present_vec[0] = 1
    cues_present_vec[1] = 1
    cues_present_vec[2] = 1

    kalman.update_covariance_matrix(covmat_matrix, cues_present_vec)

    # it should do something
    assert np.any(covmat != covmat_matrix)

    cues_present = ("a", "b", "c")
    cue_index_map = {"a": 0, "b": 1, "c": 2}
    kalman.update_covariance_loop(covmat_loop, cues_present, cue_index_map)

    # it should do something
    assert np.any(covmat != covmat_loop)

    # but both should give the same results
    print(np.sum(covmat_matrix == covmat_loop))
    print(covmat_matrix[:10, :4])
    print("-" * 50)
    print(covmat_loop[:10, :4])
    # assert np.all(covmat_matrix == covmat_loop)
