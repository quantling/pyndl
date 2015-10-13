
import numpy as np

"""
Kalman filter
=============

To implement Kalman filter [@kalman1960], we need two updating functions: for
the expectation (mean), and for the uncertainty (variance). Again, algorithm is
developed on the basis of a matrix algebra solution.

"""

def update_covariance(covmat, cues_present_vec, *, noise_parameter=1):
    """
    Update the covariance matrix ``covmat`` in place (!!).

    Parameters
    ==========

    covmat : covariance matrix of the cues (square matrix)
    cues_present_vec : a sparse column vector with ones where the cues are
            present, otherwise zero.
    noise_parameter : modulates the learning rate

    """
    if cues_present_vec.shape[0] == 1:
        raise ValueError("cues_present_vec need to be a column vector.")
    if isinstance(cues_present_vec, np.matrix)
        raise ValueError("cues_present_vec need to be a np.matrix.")

    numerator = covmat * cues_present_vec * cues_present_vec.T * covmat
    denominator = float(noise_parameter + cues_present_vec.T * covmat *
                        cues_present_vec)
    # inplace manipulation (bad behaviour, but memory efficient)
    covmat -= numerator / denominator


def update_mu(mumat, covmat, cues_present_vec, outcomes_present_vec, *,
              noise_parameter=1):
    """
    Update the mean matrix ``mumat`` in place (!!).

    Parameters
    ==========

    mumat : mean matrix connecting cues with outcomes
    covmat : covariance matrix of the cues (square matrix)
    cues_present_vec : a sparse column vector with ones where the cues are
            present, otherwise zero.
    outcomes_present_vec : a sparse column vector with ones where the outcomes
            are present, otherwise zero.
    noise_parameter : modulates the learning rate

    """

    if cues_present_vec.shape[0] == 1:
        raise ValueError("cues_present_vec need to be a column vector.")
    if isinstance(cues_present_vec, np.matrix)
        raise ValueError("cues_present_vec need to be a np.matrix.")
    if outcomes_present_vec.shape[0] == 1:
        raise ValueError("outcomes_present_vec need to be a column vector.")
    if isinstance(outcomes_present_vec, np.matrix)
        raise ValueError("outcomes_present_vec need to be a np.matrix.")

    nominator = covmat * cues_present_vec * (outcomes_present_vec -
                                             cues_present_vec.T * mumat)
    denominator = float(noise_parameter + cues_present_vec.T * covmat *
                        cues_present_vec)
    # inplace manipulation (bad behaviour, but memory efficient)
    # not memory efficient in this case
    mumat += nominator / denominator


# TODO building a cue->index map
# TODO building a outcome->index map

# TODO generating sparse column vectors out of event

# TODO expose covmat and mumat to R

