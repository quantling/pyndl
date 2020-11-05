import numpy as np
import math
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from cython.parallel cimport parallel, prange

from ndl_parallel cimport learn_inplace_ptr
from error_codes cimport ErrorCode, NO_ERROR, INITIAL_ERROR_CODE, ERROR_CODES


def learn_inplace(binary_file_paths, np.ndarray[dtype_t, ndim=2] weights,
                  dtype_t alpha, dtype_t beta1,
                  dtype_t beta2, dtype_t lambda_,
                  np.ndarray[unsigned int, ndim=1] all_outcomes,
                  unsigned int chunksize,
                  unsigned int number_of_threads):

    cdef unsigned int mm = weights.shape[1]  # number of cues == columns
    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef ErrorCode error = INITIAL_ERROR_CODE


  #  cdef String
    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *

    for binary_file_path in binary_file_paths: #
        filename_byte_string = binary_file_path.encode("UTF-8")
        fname = filename_byte_string

        number_parts = math.ceil(<double> length_all_outcomes / chunksize)

        with nogil, parallel(num_threads=number_of_threads):
            for ii in prange(number_parts, schedule="dynamic", chunksize=1):
                start_val = ii * chunksize
                end_val = min(start_val + chunksize, length_all_outcomes)
                if start_val == length_all_outcomes:
                    break
                error = learn_inplace_ptr(fname, weights_ptr, mm, alpha, beta1,
                                  beta2, lambda_, all_outcomes_ptr, start_val,
                                  end_val)
                if error != NO_ERROR:
                    break

    if (error != NO_ERROR):
        raise IOError(f'binary files does not have proper format, error code {error}\n{ERROR_CODES}')
