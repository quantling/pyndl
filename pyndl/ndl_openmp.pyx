# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
import math
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from cython.parallel cimport parallel, prange

from .ndl_parallel cimport (learn_inplace_binary_to_binary_ptr,
                            learn_inplace_binary_to_real_ptr,
                            learn_inplace_real_to_real_ptr,
                            learn_inplace_real_to_binary_ptr)
from .error_codes cimport (ErrorCode,
                           NO_ERROR,
                           MAGIC_NUMBER_DOES_NOT_MATCH,
                           VERSION_NUMBER_DOES_NOT_MATCH,
                           INITIAL_ERROR_CODE,
                           ONLY_ONE_OUTCOME_PER_EVENT,
                           ERROR_CODES)


def learn_inplace_binary_to_binary(binary_file_paths,
                  dtype_t alpha, dtype_t beta1,
                  dtype_t beta2, dtype_t lambda_,
                  np.ndarray[dtype_t, ndim=2] weights,
                  np.ndarray[unsigned int, ndim=1] all_outcomes,
                  unsigned int chunksize,
                  unsigned int n_jobs):

    cdef unsigned int n_all_cues = weights.shape[1]  # number of cues == columns
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

        with nogil, parallel(num_threads=n_jobs):
            for ii in prange(number_parts, schedule="dynamic", chunksize=1):
                start_val = ii * chunksize
                end_val = min(start_val + chunksize, length_all_outcomes)
                if start_val == length_all_outcomes:
                    break
                error = learn_inplace_binary_to_binary_ptr(fname, alpha, beta1,
                                    beta2, lambda_, weights_ptr, n_all_cues,
                                    all_outcomes_ptr, start_val, end_val)
                if error != NO_ERROR:
                    break

    if (error != NO_ERROR):
        raise IOError(f'binary files does not have proper format, error code {error}\n{ERROR_CODES}')


def learn_inplace_binary_to_real(binary_file_paths,
                                 dtype_t eta,
                                 np.ndarray[dtype_t, ndim=2] outcome_vectors,
                                 np.ndarray[dtype_t, ndim=2] weights,
                                 unsigned int chunksize,
                                 unsigned int n_jobs):

    cdef unsigned int n_all_cues = weights.shape[1]  # number of cues == columns
    cdef unsigned int n_outcome_vector_dimensions = outcome_vectors.shape[1]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef ErrorCode error = INITIAL_ERROR_CODE

    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *
    cdef dtype_t* outcome_vectors_ptr = <dtype_t *> outcome_vectors.data # ueberlegen ob [][] oder ** oder [] oder *


    for binary_file_path in binary_file_paths:
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

      number_parts = (n_outcome_vector_dimensions // chunksize)
      if n_outcome_vector_dimensions % chunksize != 0:
          number_parts += 1

      with nogil, parallel(num_threads=n_jobs):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, n_outcome_vector_dimensions)
          if start_val == n_outcome_vector_dimensions:
            break
          error = NO_ERROR
          error = learn_inplace_binary_to_real_ptr(fname,
                                                   eta,
                                                   outcome_vectors_ptr,
                                                   weights_ptr,
                                                   n_all_cues,
                                                   n_outcome_vector_dimensions,
                                                   start_val,
                                                   end_val)
    if (error == ONLY_ONE_OUTCOME_PER_EVENT):
        raise ValueError('error code %i, legal number of outcomes per event is exactly 1')
    if (error == MAGIC_NUMBER_DOES_NOT_MATCH or error == VERSION_NUMBER_DOES_NOT_MATCH):
        raise IOError('binary files does not have proper format, error code %i' % error)


def learn_inplace_real_to_binary(binary_file_paths,
                                 dtype_t beta1,
                                 dtype_t beta2,
                                 dtype_t lambda_,
                                 np.ndarray[dtype_t, ndim=2] cue_vectors,
                                 np.ndarray[dtype_t, ndim=2] weights,
                                 unsigned int chunksize,
                                 unsigned int n_jobs):

    cdef unsigned int n_all_outcomes = weights.shape[0]  # number of outcomes == rows
    cdef unsigned int n_cue_vector_dimensions = weights.shape[1]  # number of cue vector dimensions == columns
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef int error = INITIAL_ERROR_CODE

    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *
    cdef dtype_t* cue_vectors_ptr = <dtype_t *> cue_vectors.data # ueberlegen ob [][] oder ** oder [] oder *


    for binary_file_path in binary_file_paths:
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

      number_parts = (n_all_outcomes // chunksize)
      if n_all_outcomes % chunksize != 0:
          number_parts += 1

      with nogil, parallel(num_threads=n_jobs):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, n_all_outcomes)
          if start_val == n_all_outcomes:
            break
          error = NO_ERROR
          error = learn_inplace_real_to_binary_ptr(fname,
                                                   beta1,
                                                   beta2,
                                                   lambda_,
                                                   cue_vectors_ptr,
                                                   weights_ptr,
                                                   n_cue_vector_dimensions,
                                                   start_val,
                                                   end_val)
    if (error == ONLY_ONE_OUTCOME_PER_EVENT):
        raise ValueError('error code %i, legal number of outcomes per event is exactly 1')
    if (error == MAGIC_NUMBER_DOES_NOT_MATCH or error == VERSION_NUMBER_DOES_NOT_MATCH):
        raise IOError('binary files does not have proper format, error code %i' % error)


def learn_inplace_real_to_real(binary_file_paths,
                  dtype_t eta,
                  np.ndarray[dtype_t, ndim=2] cue_vectors,
                  np.ndarray[dtype_t, ndim=2] outcome_vectors,
                  np.ndarray[dtype_t, ndim=2] weights,
                  unsigned int chunksize,
                  unsigned int n_jobs):

    assert weights.shape[1] == cue_vectors.shape[1]
    assert weights.shape[0] == outcome_vectors.shape[1]

    cdef unsigned int n_cue_vector_dimensions = cue_vectors.shape[1]
    cdef unsigned int n_outcome_vector_dimensions = outcome_vectors.shape[1]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef int error = INITIAL_ERROR_CODE

    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *
    cdef dtype_t* cue_vectors_ptr = <dtype_t *> cue_vectors.data # ueberlegen ob [][] oder ** oder [] oder *
    cdef dtype_t* outcome_vectors_ptr = <dtype_t *> outcome_vectors.data # ueberlegen ob [][] oder ** oder [] oder *


    for binary_file_path in binary_file_paths:
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

      number_parts = (n_outcome_vector_dimensions // chunksize)
      if n_outcome_vector_dimensions % chunksize != 0:
          number_parts += 1

      with nogil, parallel(num_threads=n_jobs):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, n_outcome_vector_dimensions)
          if start_val == n_outcome_vector_dimensions:
            break
          error = NO_ERROR
          error = learn_inplace_real_to_real_ptr(fname,
                                                 eta,
                                                 cue_vectors_ptr,
                                                 outcome_vectors_ptr,
                                                 weights_ptr,
                                                 n_cue_vector_dimensions,
                                                 n_outcome_vector_dimensions,
                                                 start_val,
                                                 end_val)
    if (error == ONLY_ONE_OUTCOME_PER_EVENT):
        raise ValueError('error code %i, legal number of outcomes per event is exactly 1')
    if (error == MAGIC_NUMBER_DOES_NOT_MATCH or error == VERSION_NUMBER_DOES_NOT_MATCH):
        raise IOError('binary files does not have proper format, error code %i' % error)
