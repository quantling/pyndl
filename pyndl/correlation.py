import sys
import time

import numpy as np

from scipy import stats

if sys.platform.startswith('linux'):
    from . import correlation_openmp
elif sys.platform.startswith('win32'):
    pass
elif sys.platform.startswith('darwin'):
    pass


def _reference_correlation(semantics, activations, *, verbose=False):
    """
    calculates the correlations between the semantics and the activations.

    Returns
    -------
    np.array (n_outcomes, n_events)

    The first column contains all correlations between the first event and
    all possible outcomes in the semantcs.

    The first column reads like:

    0. correlation between first event and first outcome in the semantic
       (gold standard) space.
    1. correlation between first event and second outcome ...
    ...

    """
    assert semantics.shape[0] == activations.shape[0], ("number of vector dimensions in semantics and activations"
                                                        " need to be the same")
    n_outcomes = semantics.shape[1]
    n_events = activations.shape[1]

    correlations = np.zeros((n_outcomes, n_events))

    start_time = time.time()
    for ii in range(n_events):
        for jj in range(n_outcomes):
            correlations[jj, ii], _ = stats.pearsonr(semantics[:, jj], activations[:, ii])
    if verbose:
        print(f"time needed for correlations:  {time.time() - start_time}")

    return correlations


def correlation(semantics, activations, *, verbose=False, allow_nan=False):
    """
    calculates the correlations between the semantics and the activations.

    Returns
    -------
    np.array (n_outcomes, n_events)

    The first column contains all correlations between the first event and
    all possible outcomes in the semantcs.

    The first column reads like:

    0. correlation between first event and first outcome in the semantic
       (gold standard) space.
    1. correlation between first event and second outcome ...
    ...

    """
    if not sys.platform.startswith('linux'):
        raise NotImplementedError("OpenMP is linux only at the moment.")

    assert semantics.shape[0] == activations.shape[0], ("number of vector dimensions in semantics and activations"
                                                        "need to be the same")
    n_outcomes = semantics.shape[1]
    n_vec_dims, n_events = activations.shape

    semantics_means = np.zeros((n_outcomes,))
    semantics_stds = np.zeros((n_outcomes,))
    activations_means = np.zeros((n_events,))
    activations_stds = np.zeros((n_events,))

    if verbose:
        start_time = time.time()

    for jj in range(n_outcomes):
        semantics_means[jj] = np.mean(semantics[:, jj])
        semantics_stds[jj] = np.std(semantics[:, jj], ddof=1)

    for ii in range(n_events):
        activations_means[ii] = np.mean(activations[:, ii])
        activations_stds[ii] = np.std(activations[:, ii], ddof=1)

    if verbose:
        print(f"time needed for stds and means:  {time.time() - start_time}")

    if not allow_nan:
        if np.any(semantics_stds == 0) or np.any(np.isnan(semantics_stds)):
            raise ValueError('Standard deviations of semantics are not different to zero or nan.')
        if np.any(activations_stds == 0) or np.any(np.isnan(activations_stds)):
            raise ValueError('Standard deviations of activations are not different to zero or nan.')

    if verbose:
        start_time = time.time()

    correlations = correlation_openmp.correlation(semantics, activations, semantics_means,
                                                  semantics_stds, activations_means, activations_stds)
    if verbose:
        print(f"time needed for correlations:  {time.time() - start_time}")

    return correlations
