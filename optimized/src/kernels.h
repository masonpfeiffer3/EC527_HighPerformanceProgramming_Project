#ifndef KERNELS_H
#define KERNELS_H

#include "params.h"

// Existing kernels (carried over from serial unchanged)

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

// New batch kernels for

// C = A x B^T  (dot-product / horizontal-reduction form)
// A: (actual_S x k), B: (m x k), C: (actual_S x m)
// Used for forward pass and error backpropagation.
int kernel_gemm_forward(matrix_ptr A, matrix_ptr B, matrix_ptr C, int actual_S);

// dW += delta^T x act  (accumulating saxpy form)
// delta: (actual_S x out_dim), act: (actual_S x in_dim), dW: (out_dim x in_dim)
// Used for weight gradient accumulation.
int kernel_gemm_weight_grad(matrix_ptr delta, matrix_ptr act, matrix_ptr dW, int actual_S);

// Element-wise multiply: C[s][j] = A[s][j] * B[s][j]  for s in 0..actual_S.
// A, B, C must have the same shape.
int kernel_hadamard_mat(matrix_ptr A, matrix_ptr B, matrix_ptr C, int actual_S);

// Bias broadcast add: Z[s][j] += b[j]  for all s in 0..actual_S.
// Z has shape (max_S x cols), b has length cols.
int kernel_bias_broadcast_add(matrix_ptr Z, array_ptr b, int actual_S);

// Column-sum accumulate: b_sum[j] += sum_{s=0}^{actual_S-1} delta[s][j].
// delta has shape (max_S x cols), b_sum has length cols.
int kernel_bias_grad_accum(matrix_ptr delta, array_ptr b_sum, int actual_S);

// Copy first actual_S rows from src into dst.
// src and dst must have the same number of columns.
int kernel_matrix_copy_rows(matrix_ptr src, matrix_ptr dst, int actual_S);

#endif
