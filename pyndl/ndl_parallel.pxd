cimport numpy as np
ctypedef np.float64_t dtype_t
from error_codes cimport ErrorCode


cdef ErrorCode learn_inplace_ptr(char*, dtype_t*, unsigned int, dtype_t,
                        dtype_t, dtype_t, dtype_t, unsigned int*, unsigned int,
                        unsigned int) nogil

cdef ErrorCode learn_inplace_masked_ptr(char*, dtype_t*, unsigned int, dtype_t,
                        dtype_t, dtype_t, dtype_t, unsigned int*, unsigned int,
                        unsigned int) nogil
