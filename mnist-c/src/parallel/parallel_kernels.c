#include <math.h>
#include "parallel_kernels.h"

static data_t sig(data_t z) {
    return 1.0f / (1.0f + expf(-z));
}

static data_t sig_prime(data_t z) {
    data_t s = sig(z);
    return s * (1.0f - s);
}

// C = A x B
// A: [m x k], B: [k x n], C: [m x n]
// C[i,j] = sum_r A[i,r] * B[r,j]
void mat_mat_mult(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int m = get_matrix_rows(A);
    long int k = get_matrix_cols(A);
    long int n = get_matrix_cols(B);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    data_t *c = get_matrix_start(C);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data_t sum = 0.0f;
            for (int r = 0; r < k; r++) {
                sum += a[i*k + r] * b[r*n + j];
            }
            c[i*n + j] = sum;
        }
    }
}

// C = A x B^T
// A: [m x k], B: [n x k]  =>  B^T: [k x n],  C: [m x n]
// C[i,j] = sum_r A[i,r] * B[j,r]
// Used for weight gradients: dW = delta x A_prev^T
void mat_mat_mult_transB(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int m = get_matrix_rows(A);
    long int k = get_matrix_cols(A);
    long int n = get_matrix_rows(B);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    data_t *c = get_matrix_start(C);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            data_t sum = 0.0f;
            for (int r = 0; r < k; r++) {
                sum += a[i*k + r] * b[j*k + r];
            }
            c[i*n + j] = sum;
        }
    }
}

// C = A^T x B
// A: [k x m] stored row-major  =>  A^T: [m x k],  B: [k x n],  C: [m x n]
// C[i,j] = sum_r A[r,i] * B[r,j]
// Used for error propagation: err_prev = W^T x delta
void mat_mat_mult_transA(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int k = get_matrix_rows(A);
    long int m = get_matrix_cols(A);
    long int n = get_matrix_cols(B);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    data_t *c = get_matrix_start(C);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            c[i*n + j] = 0.0f;

    for (int r = 0; r < k; r++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                c[i*n + j] += a[r*m + i] * b[r*n + j];
            }
        }
    }
}

// Z[i,j] += b[i]  for all j  (bias broadcast across sample columns, in-place)
void mat_add_bias(matrix_ptr Z, array_ptr b) {
    long int rows = get_matrix_rows(Z);
    long int cols = get_matrix_cols(Z);
    data_t *z = get_matrix_start(Z);
    data_t *bv = get_array_start(b);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            z[i*cols + j] += bv[i];
        }
    }
}

// A[i,j] = sigmoid(A[i,j])  (in-place over all elements)
void mat_sigmoid_inplace(matrix_ptr A) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t *a = get_matrix_start(A);
    for (long int idx = 0; idx < total; idx++) {
        a[idx] = sig(a[idx]);
    }
}

// Out[i,j] = sigmoid_prime(Z[i,j])  (element-wise sigmoid derivative from pre-activation Z)
void mat_sigmoid_prime(matrix_ptr Z, matrix_ptr Out) {
    long int total = get_matrix_rows(Z) * get_matrix_cols(Z);
    data_t *z  = get_matrix_start(Z);
    data_t *out = get_matrix_start(Out);
    for (long int idx = 0; idx < total; idx++) {
        out[idx] = sig_prime(z[idx]);
    }
}

// C[i,j] = A[i,j] * B[i,j]  (Hadamard product, supports aliasing C==A or C==B)
void mat_hadamard(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    data_t *c = get_matrix_start(C);
    for (long int idx = 0; idx < total; idx++) {
        c[idx] = a[idx] * b[idx];
    }
}

// C[i,j] = A[i,j] - B[i,j]
void mat_sub(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    data_t *c = get_matrix_start(C);
    for (long int idx = 0; idx < total; idx++) {
        c[idx] = a[idx] - b[idx];
    }
}

// v[i] = sum_j A[i,j]  (row-wise sum; used to collapse sample dimension for bias gradients)
void mat_row_sum(matrix_ptr A, array_ptr v) {
    long int rows = get_matrix_rows(A);
    long int cols = get_matrix_cols(A);
    data_t *a  = get_matrix_start(A);
    data_t *vd = get_array_start(v);
    for (int i = 0; i < rows; i++) {
        data_t sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += a[i*cols + j];
        }
        vd[i] = sum;
    }
}

// Copy all elements of A into B (same dimensions required)
void mat_copy(matrix_ptr A, matrix_ptr B) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t *a = get_matrix_start(A);
    data_t *b = get_matrix_start(B);
    for (long int idx = 0; idx < total; idx++) {
        b[idx] = a[idx];
    }
}
