import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport *

cdef unsigned int MAGIC_NUMBER = 14159265
cdef unsigned int CURRENT_VERSION_WITH_FREQ = 215
cdef unsigned int CURRENT_VERSION = 2048 + 215

if sizeof(unsigned int) != 4:
    raise ImportError('unsigned int needs to be 4 bytes not %i bytes' % sizeof(unsigned int))


cdef inline void read_next_int(void *data, FILE *binary_file) nogil:
    fread(data, 4, 1, binary_file) # little endian


cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *) nogil
    #int fclose ( FILE * stream )
    int fclose(FILE *) nogil
    #size_t fread ( void *ptr, size_t size, size_t count, FILE *stream );
    size_t fread (void *, size_t, size_t, FILE *) nogil


def learn_inplace(binary_file_paths, np.ndarray[double, ndim=2] weights,
                  double alpha, double beta1,
                  double beta2, double lambda_,
                  np.ndarray[unsigned int, ndim=1] all_outcomes,
                  unsigned int chunksize,
                  unsigned int number_of_threads):

    cdef unsigned int mm = weights.shape[1]  # number of cues == columns
    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
    cdef char* fname
    cdef unsigned int start_val, end_val, ii, number_parts
    cdef int error = 4

  #  cdef String
    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef double* weights_ptr = <double *> weights.data # ueberlegen ob [][] oder ** oder [] oder *

    for binary_file_path in binary_file_paths: #
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

      number_parts = (length_all_outcomes // chunksize) + 1

      with nogil, parallel(num_threads=number_of_threads):
        for ii in prange(number_parts, schedule="dynamic", chunksize=1):
          start_val = ii * chunksize
          end_val = min(start_val + chunksize, length_all_outcomes)
          if start_val == length_all_outcomes:
            break
          error = 0
          error = learn_inplace_ptr(fname, weights_ptr, mm, alpha, beta1,
                            beta2, lambda_, all_outcomes_ptr, start_val,
                            end_val)
    if (error != 0):
        raise IOError('binary files does not have proper format, error code %i' % error)

def learn_inplace_2(binary_file_paths, np.ndarray[double, ndim=2] weights,
                  double alpha, double beta1,
                  double beta2, double lambda_,
                  np.ndarray[unsigned int, ndim=1] all_outcomes):

    cdef unsigned int mm = weights.shape[1]  # number of cues == columns
    cdef unsigned int* all_outcomes_ptr = <unsigned int *> all_outcomes.data
    cdef unsigned int length_all_outcomes = all_outcomes.shape[0]
    cdef char* fname
    cdef unsigned int start_val, end_val
    cdef int error = 4

  #  cdef String
    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef double* weights_ptr = <double *> weights.data # ueberlegen ob [][] oder ** oder [] oder *

    for binary_file_path in binary_file_paths: #
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

      with nogil:
          error = 0
          error = learn_inplace_ptr(fname, weights_ptr, mm, alpha, beta1,
                            beta2, lambda_, all_outcomes_ptr, 0,
                            length_all_outcomes)
    if (error != 0):
        raise IOError('binary files does not have proper format, error code %i' % error)

cdef int is_element_of(unsigned int elem, unsigned int* arr, unsigned int size) nogil:
    cdef unsigned int ii
    for ii in range(size):
      if arr[ii] == elem:
        return True
    return False


# ggf exception zurückgeben
cdef int learn_inplace_ptr(char* binary_file_path, double* weights,
                        unsigned int mm,
                        double alpha, double beta1,
                        double beta2, double lambda_,
                        unsigned int* all_outcome_indices,
                        unsigned int start,
                        unsigned int end) nogil:


    cdef unsigned int number_of_events, number_of_cues, number_of_outcomes
    cdef double association_strength, update
    cdef unsigned int magic_number, version, ii, jj, event, index, appearance
    cdef unsigned int* cue_indices
    cdef unsigned int* outcome_indices
    cdef unsigned int max_number_of_cues = 1024
    cdef unsigned int max_number_of_outcomes = 1024

    cdef FILE* cfile
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
        outcome_indices = <unsigned int *> malloc(sizeof(unsigned int) * number_of_outcomes)
        fread(outcome_indices, 4, number_of_outcomes, binary_file)

        # learn
        for ii in range(start, end):
            association_strength = 0.0
            for jj in range(number_of_cues):
              index = cue_indices[jj] + mm * all_outcome_indices[ii]
              association_strength += weights[index]
            if is_element_of(all_outcome_indices[ii], outcome_indices, number_of_outcomes):
              update = beta1 * (lambda_ - association_strength)
            else:
              update = beta2 * (0.0 - association_strength)
            for jj in range(number_of_cues):
              index = cue_indices[jj] + mm * all_outcome_indices[ii]
              weights[index] += alpha * update

    fclose(binary_file)
    free(cue_indices)
    free(outcome_indices)
    return 0
