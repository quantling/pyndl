#!/usr/bin/env python3

# pylint: disable=C0111

import pytest
import numpy as np

from pyndl import correlation


@pytest.mark.nolinux
def test_correlation():
    np.random.seed(20190507)

    n_vec_dims = 40
    n_outcomes = 50
    n_cues = 20
    n_events = 120

    semantics = np.asfortranarray(np.random.random((n_vec_dims, n_outcomes)))

    weights = np.random.random((n_vec_dims, n_cues))
    events = np.random.random((n_cues, n_events))

    # n_vec_dims x n_events
    activations = np.asfortranarray(weights @ events)

    corr1 = correlation._reference_correlation(semantics, activations, verbose=True)
    corr2 = correlation.correlation(semantics, activations, verbose=True)

    assert np.allclose(corr1, corr2)


# TODO test for inputs that are NaN or have zero std.

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
