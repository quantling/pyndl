# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

ctypedef np.float64_t dtype_t
cimport cython
from cython.parallel cimport parallel, prange


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def correlation(semantics_py,
                activations_py,
                np.ndarray[dtype_t, ndim=1] semantics_means,
                np.ndarray[dtype_t, ndim=1] semantics_stds,
                np.ndarray[dtype_t, ndim=1] activations_means,
                np.ndarray[dtype_t, ndim=1] activations_stds,
                *,
                unsigned int n_jobs=30,
                unsigned int chunksize=10):
    """
    The following formula is implemented here.

    .. math::

        r_{xy} = \frac{ \sum x_{i}y_{i}-n{\bar {x}}{\bar{y}} }{ (n-1)s_{x}s_{y} }


    """

    cdef int n_outcomes = semantics_py.shape[1]
    cdef int n_vec_dims, n_events = 0
    n_vec_dims, n_events = activations_py.shape
    cdef np.ndarray[dtype_t, ndim=2] correlations = np.zeros((n_outcomes, n_events))
    cdef int ii, jj, kk = 0
    cdef dtype_t scalar_prod, nominator, denominator = 0.0
    cdef dtype_t n_vec_dims_float = <dtype_t> n_vec_dims

    cdef np.ndarray[dtype_t, ndim=2] semantics = semantics_py
    cdef np.ndarray[dtype_t, ndim=2] activations = activations_py

    with nogil, parallel(num_threads=n_jobs):
        for ii in prange(n_events, schedule="dynamic", chunksize=chunksize):
            for jj in range(n_outcomes):
                scalar_prod = 0.0
                for kk in range(n_vec_dims):
                    scalar_prod = scalar_prod + semantics[kk, jj] * activations[kk, ii]
                nominator = scalar_prod - n_vec_dims_float * semantics_means[jj] * activations_means[ii]
                denominator = (n_vec_dims_float - 1.0) * semantics_stds[jj] * activations_stds[ii]

                correlations[jj, ii] = nominator / denominator

    return correlations

