#ifndef PARALLEL_KERNELS_H
#define PARALLEL_KERNELS_H

#include "../params.h"
#include "../serial/array_matrix_funcs.h"

// C = A x B  (standard GEMM)
// A: [m x k], B: [k x n], C: [m x n]
void mat_mat_mult(matrix_ptr A, matrix_ptr B, matrix_ptr C);

// C = A x B^T
// A: [m x k], B: [n x k] (B stored as n rows of length k), C: [m x n]
// C[i,j] = sum_r A[i,r] * B[j,r]
void mat_mat_mult_transB(matrix_ptr A, matrix_ptr B, matrix_ptr C);

// C = A^T x B
// A: [k x m] stored row-major, B: [k x n], C: [m x n]
// C[i,j] = sum_r A[r,i] * B[r,j]
void mat_mat_mult_transA(matrix_ptr A, matrix_ptr B, matrix_ptr C);

// Z[i,j] += b[i] for all j  (broadcast bias vector across columns, in-place)
void mat_add_bias(matrix_ptr Z, array_ptr b);

// A[i,j] = sigmoid(A[i,j])  (in-place)
void mat_sigmoid_inplace(matrix_ptr A);

// Out[i,j] = sigmoid_prime(Z[i,j])  (reads pre-activation Z, writes sigmoid' values)
void mat_sigmoid_prime(matrix_ptr Z, matrix_ptr Out);

// C[i,j] = A[i,j] * B[i,j]  (Hadamard / element-wise product, supports C==A)
void mat_hadamard(matrix_ptr A, matrix_ptr B, matrix_ptr C);

// C[i,j] = A[i,j] - B[i,j]
void mat_sub(matrix_ptr A, matrix_ptr B, matrix_ptr C);

// v[i] = sum_j A[i,j]  (sum each row across all columns)
void mat_row_sum(matrix_ptr A, array_ptr v);

// Copy all elements of A into B (same dimensions)
void mat_copy(matrix_ptr A, matrix_ptr B);

#endif
