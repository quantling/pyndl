# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
ctypedef np.float64_t dtype_t
from error_codes cimport ErrorCode


cdef ErrorCode learn_inplace_binary_to_binary_ptr(char*, dtype_t, dtype_t, dtype_t, dtype_t,
                        dtype_t*, unsigned int, unsigned int*, unsigned int,
                        unsigned int) noexcept nogil


cdef ErrorCode learn_inplace_binary_to_real_ptr(char*, dtype_t, dtype_t*,
                        dtype_t*, unsigned int, unsigned int, unsigned int,
                        unsigned int) noexcept nogil


cdef ErrorCode learn_inplace_real_to_real_ptr(char*, dtype_t, dtype_t*,
                        dtype_t*, dtype_t*, unsigned int, unsigned int,
                        unsigned int, unsigned int) noexcept nogil


cdef ErrorCode learn_inplace_real_to_binary_ptr(char*, dtype_t, dtype_t,
                        dtype_t, dtype_t*, dtype_t*, unsigned int, unsigned
                        int, unsigned int) noexcept nogil
