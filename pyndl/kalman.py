# !/usr/bin/env/python3
# coding: utf-8

"""
Kalman filter
-------------

To implement Kalman filter [@kalman1960], we need two updating functions: for
the expectation (mean), and for the uncertainty (variance). Again, algorithm is
developed on the basis of a matrix algebra solution.

"""

import sys

import numpy as np
from matplotlib import pyplot as plt

from .count import cues_outcomes


def update_covariance_matrix(covmat, cues_present_vec, *, noise_parameter=1):
    """
    Update the covariance matrix ``covmat`` in place (!!).

    Parameters
    ----------
    covmat : covariance matrix of the cues (square matrix)
    cues_present_vec : a sparse column vector with ones where the cues are
            present, otherwise zero.
    noise_parameter : modulates the learning rate

    """
    if not cues_present_vec.shape[1] == 1:
        raise ValueError("cues_present_vec need to be a column vector.")
    # if not isinstance(cues_present_vec, np.matrix):
    #    raise ValueError("cues_present_vec need to be a np.matrix.")

    denominator = float(noise_parameter + cues_present_vec.T * (covmat *
                                                                cues_present_vec))
    # inplace manipulation (bad behaviour, but memory efficient)
    covmat -= (covmat * cues_present_vec * cues_present_vec.T * covmat) / denominator


def update_covariance_loop(covmat, cues_present, cue_index_map, *, noise_parameter=1):
    """
    Update the covariance matrix ``covmat`` in place (!!).

    Parameters
    ----------
    covmat : covariance matrix of the cues (square matrix)
    cues_present : list of cues that are present
    cue_index_map : dict which maps the index to the cues
    noise_parameter : modulates the learning rate

    """
    idxs = [cue_index_map[cue] for cue in cues_present]
    denominator = float(noise_parameter + sum(covmat[idx, idx] for idx in idxs))
    # assuming the matrix is symmetric
    cue_cov_sum = np.sum(covmat[:, idxs], axis=1)
    for nn, value in enumerate(cue_cov_sum):
        if value == 0:
            continue
        covmat[:, nn] -= float(value) * cue_cov_sum / denominator


def update_mu_matrix(mumat, covmat, cues_present_vec, outcomes_present_vec, *,
                     noise_parameter=1):
    """
    Update the mean matrix ``mumat`` in place (!!).

    Parameters
    ----------
    mumat : mean matrix connecting cues with outcomes
    covmat : covariance matrix of the cues (square matrix)
    cues_present_vec : a sparse column vector with ones where the cues are
            present, otherwise zero.
    outcomes_present_vec : a sparse column vector with ones where the outcomes
            are present, otherwise zero.
    noise_parameter : modulates the learning rate

    """

    if not cues_present_vec.shape[1] == 1:
        raise ValueError("cues_present_vec need to be a column vector.")
    # if not isinstance(cues_present_vec, np.matrix):
    #     raise ValueError("cues_present_vec need to be a np.matrix.")
    if not outcomes_present_vec.shape[1] == 1:
        raise ValueError("outcomes_present_vec need to be a column vector.")
    # if not isinstance(outcomes_present_vec, np.matrix):
    #     raise ValueError("outcomes_present_vec need to be a np.matrix.")

    denominator = float(noise_parameter + cues_present_vec.T * covmat *
                        cues_present_vec)
    # inplace manipulation (bad behaviour, but memory efficient)
    # not memory efficient in this case

    mumat += (covmat * (cues_present_vec *
                        (outcomes_present_vec.T - cues_present_vec.T * mumat))) / denominator


def main():
    """
    A main function that runs a learning of the kalman filtering.

    .. note::

        This is here in order to get it running and might be changed heavily.

    """
    events_file = "tests/events_tiny.tab"
    n_events, cues, outcomes = cues_outcomes(events_file)

    # building a cue->index map
    tmp = list(cues.keys())
    tmp.sort()
    cue_index_map = dict(zip(tmp, range(len(tmp))))
    del tmp
    # building a outcome->index map
    tmp = list(outcomes.keys())
    tmp.sort()
    outcome_index_map = dict(zip(tmp, range(len(tmp))))
    del tmp

    # mumat has cues as rows and outcomes as columns
    mumat = np.matrix(np.zeros((len(cue_index_map), len(outcome_index_map)),
                               dtype=np.double))
    print("mumat: " + str(mumat.shape))

    # covmat has cues as rows and columns
    covmat = np.matrix(np.diag(np.ones(len(cue_index_map), dtype=np.double)))

    cue_vector = np.matrix(np.zeros((len(cue_index_map), 1)), dtype=np.double)
    outcome_vector = np.matrix(np.zeros((len(outcome_index_map), 1)),
                               dtype=np.double)

    # generating sparse column vectors out of event
    with open(events_file, "rt") as dfile:
        # skip header
        dfile.readline()

        for line in dfile:
            cues_line, outcomes_line, freq = line.split("\t")
            cues_line = cues_line.split("_")
            outcomes_line = outcomes_line.strip().split("_")
            freq = int(freq)

            for cue in cues_line:
                cue_vector[cue_index_map[cue], 0] = 1.0
            for outcome in outcomes_line:
                outcome_vector[outcome_index_map[outcome], 0] = 1.0

            # update mumat and covmat inplace
            # first update mumat as it depends on covmat
            for _ in range(freq):
                # apply dynamic change
                # mumat = D * mumat  # D is identity
                # covmat = D * covmat * D.T + U  # D is identity

                # update with next event
                update_mu_matrix(mumat, covmat, cue_vector, outcome_vector)
                update_covariance_matrix(covmat, cue_vector)

            # set values back to zero!!!
            for cue in cues_line:
                cue_vector[cue_index_map[cue]] = 0.0
            for outcome in outcomes_line:
                outcome_vector[outcome_index_map[outcome]] = 0.0
            print(".", end="")
            sys.stdout.flush()

    # TODO expose covmat and mumat to R
    np.savetxt("mumat.txt", mumat)
    np.savetxt("covmat.txt", covmat)

    plt.imshow(mumat)
    plt.show()
    plt.imshow(covmat)
    plt.show()


if __name__ == "__main__":
    main()
