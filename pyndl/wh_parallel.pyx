import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport fopen, fread, fclose, FILE

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


cdef inline void read_next_int(void *data, FILE *binary_file) nogil:
    fread(data, 4, 1, binary_file) # little endian


cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *) nogil
    #int fclose ( FILE * stream )
    int fclose(FILE *) nogil
    #size_t fread ( void *ptr, size_t size, size_t count, FILE *stream );
    size_t fread (void *, size_t, size_t, FILE *) nogil


def learn_inplace(binary_file_paths,
                  np.ndarray[dtype_t, ndim=2] outcome_vectors,
                  dtype_t eta,
                  np.ndarray[unsigned int, ndim=1] all_outcomes,
                  np.ndarray[dtype_t, ndim=2] weights,
                  unsigned int chunksize,
                  unsigned int number_of_threads):

    cdef unsigned int mm = weights.shape[1]  # number of cues == columns
    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
    cdef unsigned int n_outcome_vector_dimensions = outcome_vectors.shape[1]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef int error = 4

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

      with nogil, parallel(num_threads=number_of_threads):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, n_outcome_vector_dimensions)
          if start_val == n_outcome_vector_dimensions:
            break
          error = 0
          error = learn_inplace_binary_to_real_ptr(fname,
                                                   eta,
                                                   outcome_vectors_ptr,
                                                   weights_ptr,
                                                   mm,
                                                   n_outcome_vector_dimensions,
                                                   all_outcomes_ptr,
                                                   start_val,
                                                   end_val)
    if (error == 7):
        raise ValueError('error code %i, legal number of outcomes per event is exactly 1')
    if (error == 1 or error == 2):
        raise IOError('binary files does not have proper format, error code %i' % error)


def learn_inplace_real_to_real(binary_file_paths,
                  dtype_t eta,
                  np.ndarray[dtype_t, ndim=2] cue_vectors,
                  np.ndarray[dtype_t, ndim=2] outcome_vectors,
                  np.ndarray[dtype_t, ndim=2] weights,
                  unsigned int chunksize,
                  unsigned int number_of_threads):

    assert weights.shape[1] == cue_vectors.shape[1]
    assert weights.shape[0] == outcome_vectors.shape[1]

    cdef unsigned int n_cue_vector_dimensions = cue_vectors.shape[1]
    cdef unsigned int n_outcome_vector_dimensions = outcome_vectors.shape[1]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef int error = 4

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

      with nogil, parallel(num_threads=number_of_threads):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, n_outcome_vector_dimensions)
          if start_val == n_outcome_vector_dimensions:
            break
          error = 0
          error = learn_inplace_real_to_real_ptr(fname,
                                                 eta,
                                                 cue_vectors_ptr,
                                                 outcome_vectors_ptr,
                                                 weights_ptr,
                                                 n_cue_vector_dimensions,
                                                 n_outcome_vector_dimensions,
                                                 start_val,
                                                 end_val)
    if (error == 7):
        raise ValueError('error code %i, legal number of outcomes per event is exactly 1')
    if (error == 1 or error == 2):
        raise IOError('binary files does not have proper format, error code %i' % error)


#def learn_inplace_2(binary_file_paths, np.ndarray[dtype_t, ndim=2] weights,
#                  dtype_t alpha, dtype_t beta1,
#                  dtype_t beta2, dtype_t lambda_,
#                  np.ndarray[unsigned int, ndim=1] all_outcomes):
#
#    cdef unsigned int mm = weights.shape[1]  # number of cues == columns
#    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
#    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
#    cdef char* fname
#    cdef unsigned int start_val, end_val
#    cdef int error = 4
#
#  #  cdef String
#    # weights muss contigousarray sein und mode=c, siehe:
#    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
#    cdef dtype_t* weights_ptr = <dtype_t *> weights.data # ueberlegen ob [][] oder ** oder [] oder *
#    cdef dtype_t* outcome_vectors_ptr = <dtype_t *> outcome_vectors.data # ueberlegen ob [][] oder ** oder [] oder *
#
#    for binary_file_path in binary_file_paths: #
#      filename_byte_string = binary_file_path.encode("UTF-8")
#      fname = filename_byte_string
#
#      with nogil:
#          error = 0
#          error = learn_inplace_ptr(fname, outcome_vectors_ptr, eta,
#                  weights_ptr, mm, n_outcome_vector_dimensions, all_outcomes_ptr, 0,
#                            length_all_outcomes)
#    if (error != 0):
#        raise IOError('binary files does not have proper format, error code %i' % error)


# ggf exception zurückgeben
cdef int learn_inplace_binary_to_real_ptr(char* binary_file_path,
                        dtype_t eta,
                        dtype_t* outcome_vectors,
                        dtype_t* weights,
                        unsigned int mm,
                        unsigned int n_outcome_vector_dimensions,
                        unsigned int* all_outcome_indices,
                        unsigned int start,
                        unsigned int end) nogil:


    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef dtype_t association_strength, update, summed_outcome_vector_value
    cdef unsigned int magic_number, version, ii, jj, kk, event
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
        return 1
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return 2

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
        for ii in range(start, end):
            association_strength = 0.0
            for jj in range(number_of_cues):
                # this overflows:
                #index = cue_indices[jj] + mm * all_outcome_indices[ii]
                index = mm  # implicit cast to unsigned long long
                index *=  ii  # this can't overflow anymore
                index += cue_indices[jj]  # this can't overflow anymore
                # worst case: 4294967295 * 4294967295 + 4294967295 == 18446744069414584320 < 18446744073709551615
                association_strength += weights[index]
            summed_outcome_vector_value = 0.0
            for kk in range(number_of_outcomes):
                index = n_outcome_vector_dimensions
                index *= all_outcome_indices[kk]
                index += ii
                summed_outcome_vector_value += outcome_vectors[index]
            update = eta * (summed_outcome_vector_value - association_strength)
            for jj in range(number_of_cues):
              index = mm  # implicit cast to unsigned long long
              index *=  ii  # this can't overflow anymore
              index += cue_indices[jj]  # this can't overflow anymore
              weights[index] += update

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return 0


cdef int learn_inplace_real_to_real_ptr(char* binary_file_path,
                        dtype_t eta,
                        dtype_t* cue_vectors,
                        dtype_t* outcome_vectors,
                        dtype_t* weights,
                        unsigned int n_cue_vector_dimensions,
                        unsigned int n_outcome_vector_dimensions,
                        unsigned int start,
                        unsigned int end) nogil:


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
        return 1
    read_next_int(&version, binary_file)
    if version == CURRENT_VERSION:
        pass
    else:
        fclose(binary_file)
        return 2

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
    return 0
