#ifndef KERNELS_H
#define KERNELS_H

#include "../params.h"

int kernel_matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);
int kernel_vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out);
int kernel_matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out);

#endif
