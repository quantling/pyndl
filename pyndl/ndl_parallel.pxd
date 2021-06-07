cimport numpy as np
ctypedef np.float64_t dtype_t
from error_codes cimport ErrorCode


cdef ErrorCode learn_inplace_binary_to_binary_ptr(char*, dtype_t, dtype_t, dtype_t, dtype_t,
                        dtype_t*, unsigned int, unsigned int*, unsigned int,
                        unsigned int) nogil


cdef ErrorCode learn_inplace_binary_to_real_ptr(char*, dtype_t, dtype_t*,
                        dtype_t*, unsigned int, unsigned int, unsigned int,
                        unsigned int) nogil


cdef ErrorCode learn_inplace_real_to_real_ptr(char*, dtype_t, dtype_t*,
                        dtype_t*, dtype_t*, unsigned int, unsigned int,
                        unsigned int, unsigned int) nogil


cdef ErrorCode learn_inplace_real_to_binary_ptr(char*, dtype_t, dtype_t,
                        dtype_t, dtype_t*, dtype_t*, unsigned int, unsigned
                        int, unsigned int) nogil
