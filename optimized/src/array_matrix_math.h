#ifndef ARRAY_MATRIX_MATH_H
#define ARRAY_MATRIX_MATH_H

#include "params.h"

typedef struct {
    data_t value;
    int index;
} output_max;

// Matrix-vector operations
int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);

// Matrix-matrix operations
int matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out);
int matrix_transpose(matrix_ptr m, matrix_ptr m_out);
int matrix_scalar_mult(matrix_ptr m, data_t scalar, matrix_ptr m_out);

// Vector-vector operations
int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out);
int vector_vector_elementwise_mult(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_scalar_mult(array_ptr v1, data_t scalar, array_ptr v_out);

// Utility
int vector_copy(array_ptr source, array_ptr dest);
output_max vector_max(array_ptr v);

// Sigmoid
data_t sigmoid(data_t z);
void sigmoid_arr(array_ptr v);
data_t sigmoid_prime(data_t z);
void sigmoid_prime_arr(array_ptr v);

// Batch sigmoid: apply sigmoid / sigmoid' in-place to the first actual_S rows of m.
void sigmoid_mat(matrix_ptr m, int actual_S);
void sigmoid_prime_mat(matrix_ptr m, int actual_S);

#endif
