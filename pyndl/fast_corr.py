import time

import numpy as np

from scipy import stats
# from numba import jit

# from corr_parallel import low_level_corr on linux
if sys.platform.startswith('linux'):
    from . import corr_parallel
elif sys.platform.startswith('win32'):
    pass
elif sys.platform.startswith('darwin'):
    pass


# np.random.seed(20190507)

# n_vec_dims = 4000
# n_outcomes = 5000
# n_cues = 40000
# n_events = 1200

# semantics = np.asfortranarray(np.random.random((n_vec_dims, n_outcomes)))

# weights = np.random.random((n_vec_dims, n_cues))
# events = np.random.random((n_cues, n_events))


# n_vec_dims x n_events
# activations = np.asfortranarray(weights @ events)


def reference_corr(semantics, activations):
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
    print(f"time needed for correlations:  {time.time() - start_time}")

    return correlations


def manual_corr(semantics, activations):
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

    start_time = time.time()
    for jj in range(n_outcomes):
        semantics_means[jj] = np.mean(semantics[:, jj])
        semantics_stds[jj] = np.std(semantics[:, jj], ddof=1)

    for ii in range(n_events):
        activations_means[ii] = np.mean(activations[:, ii])
        activations_stds[ii] = np.std(activations[:, ii], ddof=1)
    print(f"time needed for stds and means:  {time.time() - start_time}")

    start_time = time.time()
    correlations = corr_parallel.low_level_corr(semantics, activations, semantics_means,
                                                semantics_stds, activations_means, activations_stds)
    print(f"time needed for correlations:  {time.time() - start_time}")

    return correlations

# if __name__=="__main__":
#     reference_corr(semantics, activations)
#     manual_corr(semantics, activations)

# @jit(nopython=True)

# %time X2 = reference_corr(semantics, activations)
# CPU times: user 1min 11s, sys: 0 ns, total: 1min 11s
# Wall time: 1min 11s

# without numba jit
# %time X = manual_corr(semantics, activations)
# time needed for stds and means:  0.5793533325195312
# time needed for correlations:  19.464678525924683
# CPU times: user 20 s, sys: 0 ns, total: 20 s
# Wall time: 20 s

# np.allclose(X, X2)

# events 120
# time needed for correlations:  57.368293046951294
# time needed for stds and means:  0.3004155158996582
# time needed for correlations:  0.8774964809417725


# cython - events 1200

# %time X = manual_corr(semantics, activations)
# time needed for stds and means:  0.19872736930847168
# time needed for correlations:  4.957022190093994
# CPU times: user 29.9 s, sys: 32 ms, total: 29.9 s
# Wall time: 5.16 s

# %time X2 = reference_corr(semantics, activations)
# CPU times: user 5min 32s, sys: 20 ms, total: 5min 32s
# Wall time: 5min 32s

# cython - events 12000

# %time X = manual_corr(semantics, activations)
# time needed for stds and means:  0.556868314743042
# time needed for correlations:  56.93205285072327
# CPU times: user 5min 41s, sys: 392 ms, total: 5min 42s
# Wall time: 57.5 s

# import imp
# imp.reload(low_level_corr)


# python setup.py build_ext --inplace
# %run
