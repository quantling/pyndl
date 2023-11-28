# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
import math
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport fopen, fread, fclose, FILE

from error_codes cimport ErrorCode, NO_ERROR, MAGIC_NUMBER_DOES_NOT_MATCH, VERSION_NUMBER_DOES_NOT_MATCH, INITIAL_ERROR_CODE, ERROR_CODES


cdef unsigned int MAGIC_NUMBER = 14159265
cdef unsigned int CURRENT_VERSION_WITH_FREQ = 215
cdef unsigned int CURRENT_VERSION = 2048 + 215


# run two sanity checks while loading the extension
# 1. check
if sizeof(unsigned int) != 4:
    raise ImportError('unsigned int needs to be 4 bytes not %i bytes' % sizeof(unsigned int))

# 2. check
# integer overflow in uni-dimensional index
cdef unsigned long long test_index
cdef unsigned int test_cue_index, test_mm, test_outcome_index
test_cue_index = test_mm = test_outcome_index = 4294967295

test_index = test_mm
test_index *= test_outcome_index
test_index += test_cue_index
assert test_index == 18446744069414584320


cdef inline void read_next_int(void *data, FILE *binary_file) noexcept nogil:
    fread(data, 4, 1, binary_file) # little endian


cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *) noexcept nogil
    #int fclose ( FILE * stream )
    int fclose(FILE *) noexcept nogil
    #size_t fread ( void *ptr, size_t size, size_t count, FILE *stream );
    size_t fread (void *, size_t, size_t, FILE *) noexcept nogil


def learn_inplace_binary_to_binary(binary_file_paths,
                  dtype_t alpha, dtype_t beta1,
                  dtype_t beta2, dtype_t lambda_,
                  np.ndarray[dtype_t, ndim=2] weights,
                  np.ndarray[unsigned int, ndim=1] all_outcomes):

    cdef unsigned int n_all_cues = weights.shape[1]  # number of cues == columns
    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
    cdef char* fname
    cdef unsigned int start_val, end_val
    cdef ErrorCode error = INITIAL_ERROR_CODE

  #  cdef String
    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *

    for binary_file_path in binary_file_paths: #
        filename_byte_string = binary_file_path.encode("UTF-8")
        fname = filename_byte_string

        with nogil:
            error = learn_inplace_binary_to_binary_ptr(fname, alpha, beta1, beta2, lambda_,
                              weights_ptr, n_all_cues, all_outcomes_ptr, 0,
                              length_all_outcomes)
            if error != NO_ERROR:
                break

    if (error != NO_ERROR):
        raise IOError(f'binary files does not have proper format, error code {error}\n{ERROR_CODES}')



cdef int is_element_of(unsigned int elem, unsigned int* arr, unsigned int size) noexcept nogil:
    cdef unsigned int ii
    for ii in range(size):
        if arr[ii] == elem:
            return True
    return False


# ggf exception zurückgeben
cdef ErrorCode learn_inplace_binary_to_binary_ptr(char* binary_file_path,
                        dtype_t alpha, dtype_t beta1,
                        dtype_t beta2, dtype_t lambda_,
                        dtype_t* weights,
                        unsigned int n_all_cues,
                        unsigned int* all_outcome_indices,
                        unsigned int start,
                        unsigned int end) noexcept nogil:


    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef dtype_t association_strength, update
    cdef unsigned int magic_number, version, ii, jj, event, appearance
    cdef unsigned long long index
    cdef unsigned int* cue_indices
    cdef unsigned int* outcome_indices
    cdef unsigned int max_number_of_cues = 1024
    cdef unsigned int max_number_of_outcomes = 1024

    cdef FILE* binary_file
    binary_file = fopen(binary_file_path, "rb")

    read_next_int(&magic_number, binary_file)
    if not magic_number == MAGIC_NUMBER:
        fclose(binary_file)
        return MAGIC_NUMBER_DOES_NOT_MATCH
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return VERSION_NUMBER_DOES_NOT_MATCH

    # preallocate memory
    cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
    outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)

    read_next_int(&number_of_events, binary_file)

    for event in range(number_of_events):
        # cues
        read_next_int(&number_of_cues, binary_file)
        if number_of_cues > max_number_of_cues:
            max_number_of_cues = number_of_cues
            free(cue_indices)
            cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
        fread(cue_indices, 4, number_of_cues, binary_file)

        # outcomes
        read_next_int(&number_of_outcomes, binary_file)
        if number_of_outcomes > max_number_of_outcomes:
            max_number_of_outcomes = number_of_outcomes
            free(outcome_indices)
            outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)
        fread(outcome_indices, 4, number_of_outcomes, binary_file)

        # learn
        for ii in range(start, end):
            association_strength = 0.0
            for jj in range(number_of_cues):
                # this overflows:
                #index = cue_indices[jj] + mm * all_outcome_indices[ii]
                index = n_all_cues  # implicit cast to unsigned long long
                index *=  all_outcome_indices[ii]  # this can't overflow anymore
                index += cue_indices[jj]  # this can't overflow anymore
                # worst case: 4294967295 * 4294967295 + 4294967295 == 18446744069414584320 < 18446744073709551615
                association_strength += weights[index]
            if is_element_of(all_outcome_indices[ii], outcome_indices, number_of_outcomes):
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0.0 - association_strength)
            for jj in range(number_of_cues):
                index = n_all_cues  # implicit cast to unsigned long long
                index *=  all_outcome_indices[ii]  # this can't overflow anymore
                index += cue_indices[jj]  # this can't overflow anymore
                weights[index] += alpha * update

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return NO_ERROR


# ggf exception zurückgeben
cdef ErrorCode learn_inplace_binary_to_real_ptr(char* binary_file_path,
                        dtype_t eta,
                        dtype_t* outcome_vectors,
                        dtype_t* weights,
                        unsigned int n_all_cues,
                        unsigned int n_outcome_vector_dimensions,
                        unsigned int start,
                        unsigned int end) noexcept nogil:


    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef dtype_t association_strength, update, summed_outcome_vector_value
    cdef unsigned int magic_number, version, outcome_vec_dim_ii, jj, kk, event
    cdef unsigned long long index
    cdef unsigned int* cue_indices
    cdef unsigned int* outcome_indices
    cdef unsigned int max_number_of_cues = 1024
    cdef unsigned int max_number_of_outcomes = 1024

    cdef FILE* binary_file
    binary_file = fopen(binary_file_path, "rb")

    read_next_int(&magic_number, binary_file)
    if not magic_number == MAGIC_NUMBER:
        fclose(binary_file)
        return MAGIC_NUMBER_DOES_NOT_MATCH
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return VERSION_NUMBER_DOES_NOT_MATCH

    # preallocate memory
    cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
    outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)

    read_next_int(&number_of_events, binary_file)

    for event in range(number_of_events):
        # cues
        read_next_int(&number_of_cues, binary_file)
        if number_of_cues > max_number_of_cues:
            max_number_of_cues = number_of_cues
            free(cue_indices)
            cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
        fread(cue_indices, 4, number_of_cues, binary_file)

        # outcomes
        read_next_int(&number_of_outcomes, binary_file)
        if number_of_outcomes > max_number_of_outcomes:
            max_number_of_outcomes = number_of_outcomes
            free(outcome_indices)
            outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)
        fread(outcome_indices, 4, number_of_outcomes, binary_file)

        # learn
        for outcome_vec_dim_ii in range(start, end):
            association_strength = 0.0
            for jj in range(number_of_cues):
                # this overflows:
                #index = cue_indices[jj] + n_all_cues * outcome_vec_dim_ii
                index = n_all_cues  # implicit cast to unsigned long long
                index *= outcome_vec_dim_ii  # this can't overflow anymore
                index += cue_indices[jj]  # this can't overflow anymore
                # worst case: 4294967295 * 4294967295 + 4294967295 == 18446744069414584320 < 18446744073709551615
                association_strength += weights[index]
            summed_outcome_vector_value = 0.0
            for kk in range(number_of_outcomes):
                index = n_outcome_vector_dimensions
                index *= outcome_indices[kk]
                index += outcome_vec_dim_ii
                summed_outcome_vector_value += outcome_vectors[index]
            update = eta * (summed_outcome_vector_value - association_strength)
            for jj in range(number_of_cues):
              index = n_all_cues  # implicit cast to unsigned long long
              index *=  outcome_vec_dim_ii  # this can't overflow anymore
              index += cue_indices[jj]  # this can't overflow anymore
              weights[index] += update

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return NO_ERROR


cdef ErrorCode learn_inplace_real_to_real_ptr(char* binary_file_path,
                        dtype_t eta,
                        dtype_t* cue_vectors,
                        dtype_t* outcome_vectors,
                        dtype_t* weights,
                        unsigned int n_cue_vector_dimensions,
                        unsigned int n_outcome_vector_dimensions,
                        unsigned int start,
                        unsigned int end) noexcept nogil:

    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef dtype_t association_strength, update, summed_cue_vector_value, summed_outcome_vector_value
    cdef unsigned int magic_number, version, ii, jj, kk, event, outcome_vec_dim_ii
    cdef unsigned long long index, index_cue, index_weight
    cdef unsigned int* cue_indices
    cdef unsigned int* outcome_indices
    cdef unsigned int max_number_of_cues = 1024
    cdef unsigned int max_number_of_outcomes = 1024

    cdef FILE* binary_file
    binary_file = fopen(binary_file_path, "rb")

    read_next_int(&magic_number, binary_file)
    if not magic_number == MAGIC_NUMBER:
        fclose(binary_file)
        return MAGIC_NUMBER_DOES_NOT_MATCH
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return VERSION_NUMBER_DOES_NOT_MATCH

    # preallocate memory
    cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
    outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)

    read_next_int(&number_of_events, binary_file)

    for event in range(number_of_events):
        # cues
        read_next_int(&number_of_cues, binary_file)
        if number_of_cues > max_number_of_cues:
            max_number_of_cues = number_of_cues
            free(cue_indices)
            cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
        fread(cue_indices, 4, number_of_cues, binary_file)

        # outcomes
        read_next_int(&number_of_outcomes, binary_file)
        if number_of_outcomes > max_number_of_outcomes:
            max_number_of_outcomes = number_of_outcomes
            free(outcome_indices)
            outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)
        fread(outcome_indices, 4, number_of_outcomes, binary_file)

        # learn
        # start and end are refering to the vector dimensions of the outcome vector not to the outcomes
        for outcome_vec_dim_ii in range(start, end):
            ## W[ii, :] @ cue_vectors[jj] + W[ii, :] @ cue_vectors[jj]  + ...
            #association_strength = 0.0
            #for jj in range(number_of_cues):
            #    index_weight = n_cue_vector_dimensions
            #    index_weight *= outcome_vec_dim_ii
            #    index_weight += kk
            #    for kk in range(n_cue_vector_dimensions):
            #        index_cue = n_cue_vector_dimensions
            #        index_cue *= all_cue_indices[jj]
            #        index_cue += kk
            #        association_strength += cue_vectors[index_cue] * weights[index_weight]
            # = W[ii, :] @ (cue_vectors[jj] + cue_vectors[jj] + ...)
            association_strength = 0.0
            for kk in range(n_cue_vector_dimensions):
                summed_cue_vector_value = 0.0
                for jj in range(number_of_cues):
                    index_cue = n_cue_vector_dimensions
                    index_cue *= cue_indices[jj]
                    index_cue += kk
                    summed_cue_vector_value += cue_vectors[index_cue]
                index_weight = n_cue_vector_dimensions
                index_weight *= outcome_vec_dim_ii
                index_weight += kk
                association_strength += summed_cue_vector_value * weights[index_weight]

            summed_outcome_vector_value = 0.0
            for jj in range(number_of_outcomes):
                index = n_outcome_vector_dimensions
                index *= outcome_indices[jj]
                index += outcome_vec_dim_ii
                summed_outcome_vector_value += outcome_vectors[index]
            # update = prediction error in learning * learning rate
            update = eta * (summed_outcome_vector_value - association_strength)

            for kk in range(n_cue_vector_dimensions):
                summed_cue_vector_value = 0.0
                for jj in range(number_of_cues):
                    index_cue = n_cue_vector_dimensions
                    index_cue *= cue_indices[jj]
                    index_cue += kk
                    summed_cue_vector_value += cue_vectors[index_cue]
                index_weight = n_cue_vector_dimensions
                index_weight *= outcome_vec_dim_ii
                index_weight += kk
                weights[index_weight] += update * summed_cue_vector_value

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return NO_ERROR


cdef ErrorCode learn_inplace_real_to_binary_ptr(char* binary_file_path,
                        dtype_t beta1,
                        dtype_t beta2,
                        dtype_t lambda_,
                        dtype_t* cue_vectors,
                        dtype_t* weights,
                        unsigned int n_cue_vector_dimensions,
                        unsigned int start,
                        unsigned int end) noexcept nogil:

    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef dtype_t association_strength, update, summed_cue_vector_value
    cdef unsigned int magic_number, version, ii, jj, kk, event
    cdef unsigned long long index, index_cue, index_weight
    cdef unsigned int* cue_indices
    cdef unsigned int* outcome_indices
    cdef unsigned int max_number_of_cues = 1024
    cdef unsigned int max_number_of_outcomes = 1024

    cdef FILE* binary_file
    binary_file = fopen(binary_file_path, "rb")

    read_next_int(&magic_number, binary_file)
    if not magic_number == MAGIC_NUMBER:
        fclose(binary_file)
        return MAGIC_NUMBER_DOES_NOT_MATCH
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return VERSION_NUMBER_DOES_NOT_MATCH

    # preallocate memory
    cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
    outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)

    read_next_int(&number_of_events, binary_file)

    for event in range(number_of_events):
        # cues
        read_next_int(&number_of_cues, binary_file)
        if number_of_cues > max_number_of_cues:
            max_number_of_cues = number_of_cues
            free(cue_indices)
            cue_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_cues)
        fread(cue_indices, 4, number_of_cues, binary_file)

        # outcomes
        read_next_int(&number_of_outcomes, binary_file)
        if number_of_outcomes > max_number_of_outcomes:
            max_number_of_outcomes = number_of_outcomes
            free(outcome_indices)
            outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * max_number_of_outcomes)
        fread(outcome_indices, 4, number_of_outcomes, binary_file)

        # learn
        for ii in range(start, end):
            ## W[ii, :] @ cue_vectors[jj] + W[ii, :] @ cue_vectors[jj]  + ...
            #association_strength = 0.0
            #for jj in range(number_of_cues):
            #    index_weight = n_cue_vector_dimensions
            #    index_weight *= outcome_vec_dim_ii
            #    index_weight += kk
            #    for kk in range(n_cue_vector_dimensions):
            #        index_cue = n_cue_vector_dimensions
            #        index_cue *= all_cue_indices[jj]
            #        index_cue += kk
            #        association_strength += cue_vectors[index_cue] * weights[index_weight]
            # = W[ii, :] @ (cue_vectors[jj] + cue_vectors[jj] + ...)
            association_strength = 0.0
            for kk in range(n_cue_vector_dimensions):
                summed_cue_vector_value = 0.0
                for jj in range(number_of_cues):
                    index_cue = n_cue_vector_dimensions
                    index_cue *= cue_indices[jj]
                    index_cue += kk
                    summed_cue_vector_value += cue_vectors[index_cue]
                index_weight = n_cue_vector_dimensions
                index_weight *= ii
                index_weight += kk
                association_strength += summed_cue_vector_value * weights[index_weight]

            # update = prediction error in learning * learning rate
            if is_element_of(ii, outcome_indices, number_of_outcomes):
                update = beta1 * (lambda_ - association_strength)
            else:
                update = beta2 * (0.0 - association_strength)

            for kk in range(n_cue_vector_dimensions):
                summed_cue_vector_value = 0.0
                for jj in range(number_of_cues):
                    index_cue = n_cue_vector_dimensions
                    index_cue *= cue_indices[jj]
                    index_cue += kk
                    summed_cue_vector_value += cue_vectors[index_cue]
                index_weight = n_cue_vector_dimensions
                index_weight *= ii
                index_weight += kk
                weights[index_weight] += update * summed_cue_vector_value

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return NO_ERROR
