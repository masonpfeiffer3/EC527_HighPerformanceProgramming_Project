#ifndef KERNELS_H
#define KERNELS_H

#include "../params.h"

int kernel_matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);
int kernel_vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out);
int kernel_matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out);

int kernel_vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out);

int kernel_matrix_transpose(matrix_ptr m, matrix_ptr m_out);
int kernel_matrix_scalar_mult(matrix_ptr m, data_t scalar, matrix_ptr m_out);
int kernel_vector_scalar_mult(array_ptr v1, data_t scalar, array_ptr v_out);

void kernel_sigmoid_arr(array_ptr v);

int kernel_matrix_saxpy(matrix_ptr grad, data_t scale, matrix_ptr W);
int kernel_vector_saxpy(array_ptr grad, data_t scale, array_ptr b);

#endif
