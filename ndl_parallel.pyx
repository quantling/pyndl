import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange
from libc.stdlib cimport abort, malloc, free
from libc.stdio cimport *

cdef unsigned long MAGIC_NUMBER = 14159265
cdef unsigned long CURRENT_VERSION_WITH_FREQ = 215
cdef unsigned long CURRENT_VERSION = 2048 + 215

cdef bytes to_bytes(int_):
    return int_.to_bytes(4, 'little')
#    cdef bytes byte_ = (bytes) int_
#    return byte_

cdef int to_integer(byte_):
    return int.from_bytes(byte_, "little")
#    cdef int int_ = (int) byte_
#    return int_

cdef void read_next_long(void *data, FILE *binary_file):
    fread(data, 4, 1, binary_file) # little endian


#cdef int read_next_integer(binary_file):
#    return int.from_bytes(binary_file.read(4), "little")

cdef extern from "stdio.h":
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)
    #size_t fread ( void *ptr, size_t size, size_t count, FILE *stream );
    size_t fread ( void *, size_t, size_t, FILE *)

def learn_inplace(binary_file_paths, np.ndarray[double, ndim=2] weights,
                  double alpha, double beta1,
                  double beta2, double lambda_,
                  np.ndarray[unsigned long, ndim=1] all_outcomes,
                  unsigned long chunksize,
                  unsigned int number_of_threads):

    cdef unsigned long n = weights.shape[0]
    cdef unsigned long m = weights.shape[1]
#    cdef double alpha = alpha
    # cdef double beta1 = beta1
    # cdef double beta2 = beta2
    # cdef double lambda_ = lambda_
    cdef unsigned long* all_outcomes_ptr = <unsigned long *> all_outcomes.data
    cdef unsigned long length_all_outcomes = all_outcomes.shape[0]
    cdef char* fname
    cdef unsigned long start_val

  #  cdef String
    # weights muss contigousarray sein und mode=c, siehe:
    #cdef np.ndarray[np.uint32_t, ndim=3, mode = 'c'] np_buff = np.ascontiguousarray(im, dtype = np.uint32)
    cdef double* weights_ptr = <double *> weights.data # ueberlegen ob [][] oder ** oder [] oder *

    for binary_file_path in binary_file_paths: #
      filename_byte_string = binary_file_path.encode("UTF-8")
      fname = filename_byte_string

#      with nogil, parallel():
      for ii in range(int(length_all_outcomes / chunksize) + 1 ):
        start_val = ii * chunksize
        end_val = min(start_val + chunksize, length_all_outcomes)
        if start_val == length_all_outcomes:
          break
        learn_inplace_ptr(fname, weights_ptr, n, m, alpha, beta1,
                          beta2, lambda_, all_outcomes_ptr, start_val,
                          end_val)

cdef int is_element_of(unsigned long elem, unsigned long* arr, unsigned long size):
    for ii in range(size):
      if arr[ii] == elem:
        return True
    return False

# ggf exception zurückgeben
cdef int learn_inplace_ptr(char* binary_file_path, double* weights,
                        unsigned long n, unsigned long m,
                        double alpha, double beta1,
                        double beta2, double lambda_,
                        unsigned long* all_outcome_indices,
                        unsigned long start,
                        unsigned long end):


    cdef unsigned long number_of_events, number_of_cues, number_of_outcomes, frequency_counter
    cdef double association_strength, update
#    cdef np.ndarray[long] cue_indices, outcome_indices
    cdef unsigned long magic_number, version, ii, jj
    cdef unsigned long* cue_indices
    cdef unsigned long* outcome_indices
    cdef int has_frequency # should be changed to an equivalent of boolean

    cdef FILE* cfile
    binary_file = fopen(binary_file_path, "rb")

    read_next_long(&magic_number, binary_file)
    if not magic_number == MAGIC_NUMBER:
      return 1
    read_next_long(&version, binary_file)
    if version == CURRENT_VERSION_WITH_FREQ:
        has_frequency = True
    elif version == CURRENT_VERSION:
        has_frequency = False
    else:
      weights[0] = version
  #    weights[1] = CURRENT_VERSION
      return 2

    read_next_long(&number_of_events, binary_file)

    if not has_frequency:
      for event in range(number_of_events):

        # Cues
        read_next_long(&number_of_cues, binary_file)
        cue_indices = <unsigned long *> malloc(sizeof(unsigned long) * number_of_cues)
        for ii in range(number_of_cues):
          read_next_long(&cue_indices[ii], binary_file)

        # outcomes
        read_next_long(&number_of_outcomes, binary_file)
        outcome_indices = <unsigned long *> malloc(sizeof(unsigned long) * number_of_outcomes)
        for jj in range(number_of_outcomes):
          read_next_long(&outcome_indices[jj], binary_file)

        for jj in range(start, end):
            association_strength = 0.0
            for ii in range(number_of_cues):
              association_strength += weights[m * ii + jj]
            if is_element_of(all_outcome_indices[ii], outcome_indices, number_of_outcomes):
              update = beta1 * (lambda_ - association_strength)
            else:
              update = beta2 * (0.0 - association_strength)
            for ii in range(number_of_cues):
              weights[0] = 1.0
              #weights[all_outcome_indices * ii ][cue_indices * jj] += alpha * update
              # weights[0] += alpha * update
              #pass
               #               weights[m * ii + jj] += alpha * update

        free(cue_indices)
        free(outcome_indices)
    else:
      return 3

    fclose(binary_file)
    return 0
    ##################################################################


        # else:
        #     for event in range(nr_of_events):
        #         # Cues
        #         number_of_cues = read_next_integer(binary_file)
        #         cue_indices = np.array([read_next_integer(binary_file)
        #                             for ii in range(number_of_cues)], dtype=int) # evtl gleich in _update_... indizieren
        #         # outcomes
        #         number_of_outcomes = read_next_integer(binary_file)
        #         outcome_indices = np.array([read_next_integer(binary_file)
        #                                 for ii in range(number_of_outcomes)],
        #                                 dtype=int)
        #         # frequency
        #         frequency_counter = read_next_integer(binary_file)
        #         for appearance in range(frequency_counter):
        #             for outcome_index in all_outcome_indices:
        #                 association_strength = 0.0
        #                 for cue_index in cue_indices:
        #                     association_strength += weights[outcome_index][cue_index]
        #                 if outcome_index in outcome_indices:
        #                     update = beta1 * (lambda_ - association_strength)
        #                 else:
        #                     update = beta2 * (0.0 - association_strength)
        #                 for cue_index in cue_indices:
        #                     weights[outcome_index][cue_index] += alpha * update



def slice_list(li, sequence):
    """
    Slices a list in sublists with the length sequence.

    Parameters
    ==========
    li : list
         list which should be sliced in sublists
    sequence : int
         integer which determines the length of the sublists

    Returns
    =======
    seq_list : list of lists
        a list of sublists with the length sequence

    """
    assert len(li) == len(set(li))
    ii = 0
    seq_list = list()
    while ii < len(li):
        seq_list.append(li[ii:ii+sequence])
        ii = ii+sequence

    return seq_list
