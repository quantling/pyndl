import numpy as np
cimport numpy as np
cimport cython

################
## Preprocessing
################

MAGIC_NUMBER = 14159265
CURRENT_VERSION_WITH_FREQ = 215
CURRENT_VERSION = 2048 + 215


cdef bytes to_bytes(int_):
    return int_.to_bytes(4, 'little')
#    cdef bytes byte_ = (bytes) int_
#    return byte_

cdef int to_integer(byte_):
    return int.from_bytes(byte_, "little")
#    cdef int int_ = (int) byte_
#    return int_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _update_binary_events_inplace(np.ndarray[int, ndim=2] event_list,
                                  np.ndarray[double, ndim=2] weights,
                                  np.ndarray[long] all_outcome_indices,
                                  cython.double alpha, cython.double beta1, 
                                  cython.double beta2, double lambda_):

    for cue_indices, outcome_indices in event_list:
        _update_numpy_array_inplace(weights,
                                    cue_indices,
                                    outcome_indices,
                                    all_outcome_indices,
                                    alpha, beta1, beta2, lambda_)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _update_numpy_array_inplace(np.ndarray[double, ndim=2] weights,
                                      np.ndarray[long] cue_indices,
                                      np.ndarray[long] outcome_indices,
                                      np.ndarray[long] all_outcome_indices,
                                      cython.double alpha, cython.double beta1,
                                      cython.double beta2, double lambda_):
    cdef double association_strength = 0.0
    cdef double update = 0.0
    for outcome_index in all_outcome_indices:
        association_strength = 0.0
        for cue_index in cue_indices:
            association_strength += weights[outcome_index][cue_index]
        if outcome_index in outcome_indices:
            update = beta1 * (lambda_ - association_strength)
        else:
            update = beta2 * (0.0 - association_strength)
        for cue_index in cue_indices:
            weights[outcome_index][cue_index] += alpha * update


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def learn_inplace(binary_file_path, np.ndarray[double, ndim=2] weights,
                        cython.double alpha, cython.double beta1, 
                        cython.double beta2, cython.double lambda_, 
                        np.ndarray[long] all_outcome_indices):

    with open(binary_file_path, "rb") as binary_file:
        magic_number = to_integer(binary_file.read(4))
        if not magic_number == MAGIC_NUMBER:
            raise ValueError('Header does not match the magic number')
        version = to_integer(binary_file.read(4))
        if version == CURRENT_VERSION_WITH_FREQ:
            frequency = True
        elif version == CURRENT_VERSION:
            frequency = False
        else:
            raise ValueError('Version is incorrectly specified')

        nr_of_events = to_integer(binary_file.read(4))
        if not frequency:
            for event in range(nr_of_events):
                # Cues
                number_of_cues = to_integer(binary_file.read(4)) # cdef int indirekt casten
                cue_ids = np.array([to_integer(binary_file.read(4))
                                    for ii in range(number_of_cues)], dtype=int) # evtl gleich in _update_... indizieren
                # outcomes
                number_of_outcomes = to_integer(binary_file.read(4))
                outcome_ids = np.array([to_integer(binary_file.read(4))
                                        for ii in range(number_of_outcomes)],
                                        dtype=int)
                _update_numpy_array_inplace(weights, cue_ids,
                                            outcome_ids,
                                            all_outcome_indices, alpha, beta1,
                                            beta2, lambda_) # Code direkt reinkopieren

        else:
            for event in range(nr_of_events):
                #cues
                number_of_cues = to_integer(binary_file.read(4))
                cue_ids = np.array([to_integer(binary_file.read(4))
                                    for ii in range(number_of_cues)], dtype=int)
                #outcomes
                number_of_outcomes = to_integer(binary_file.read(4))
                outcome_ids = np.array([to_integer(binary_file.read(4))
                                        for ii in range(number_of_outcomes)],
                                        dtype=int)
                # frequency
                frequency_counter = to_integer(binary_file.read(4))
                for appearance in range(frequency_counter):
                    _update_numpy_array_inplace(weights, cue_ids,
                                                outcome_ids,
                                                all_outcome_indices, alpha,
                                                beta1, beta2, lambda_)
