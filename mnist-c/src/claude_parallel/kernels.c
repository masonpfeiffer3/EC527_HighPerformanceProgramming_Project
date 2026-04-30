#include "kernels.h"
#include "array_matrix_funcs.h"
#include <immintrin.h>          /* AVX, AVX2, FMA, SSE3 intrinsics      */
#include <string.h>

#define AVX_STRIDE   8          /* floats per 256-bit register           */
#define AVX_STRIDE_6 48         /* 6 * AVX_STRIDE: floats per unrolled iter */

// =====================================================================
// Existing kernels — carried over from serial unchanged
// =====================================================================

int kernel_matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out) {

    long int rows = get_matrix_rows(m);
    long int cols = get_matrix_cols(m);
    long int vlen = get_array_length(v);

    data_t* restrict weights              = get_matrix_start(m);
    data_t* restrict lastLayerActivations = get_array_start(v);
    data_t* restrict v_out_loc            = get_array_start(v_out);

    if (vlen == cols) {
        for (int i = 0; i < rows; i++) {
            long int row_offset = (long int)i * cols;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();

            int j;

            for (j = 0; j < cols - (AVX_STRIDE_6 - 1); j += AVX_STRIDE_6) {
                __m256 w0 = _mm256_loadu_ps(&weights[row_offset + j]);
                __m256 a0 = _mm256_loadu_ps(&lastLayerActivations[j]);
                acc0 = _mm256_fmadd_ps(w0, a0, acc0);

                __m256 w1 = _mm256_loadu_ps(&weights[row_offset + j + AVX_STRIDE]);
                __m256 a1 = _mm256_loadu_ps(&lastLayerActivations[j + AVX_STRIDE]);
                acc1 = _mm256_fmadd_ps(w1, a1, acc1);

                __m256 w2 = _mm256_loadu_ps(&weights[row_offset + j + 2*AVX_STRIDE]);
                __m256 a2 = _mm256_loadu_ps(&lastLayerActivations[j + 2*AVX_STRIDE]);
                acc2 = _mm256_fmadd_ps(w2, a2, acc2);

                __m256 w3 = _mm256_loadu_ps(&weights[row_offset + j + 3*AVX_STRIDE]);
                __m256 a3 = _mm256_loadu_ps(&lastLayerActivations[j + 3*AVX_STRIDE]);
                acc3 = _mm256_fmadd_ps(w3, a3, acc3);

                __m256 w4 = _mm256_loadu_ps(&weights[row_offset + j + 4*AVX_STRIDE]);
                __m256 a4 = _mm256_loadu_ps(&lastLayerActivations[j + 4*AVX_STRIDE]);
                acc4 = _mm256_fmadd_ps(w4, a4, acc4);

                __m256 w5 = _mm256_loadu_ps(&weights[row_offset + j + 5*AVX_STRIDE]);
                __m256 a5 = _mm256_loadu_ps(&lastLayerActivations[j + 5*AVX_STRIDE]);
                acc5 = _mm256_fmadd_ps(w5, a5, acc5);
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc0 = _mm256_add_ps(acc0, acc4);

            for (; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 w = _mm256_loadu_ps(&weights[row_offset + j]);
                __m256 a = _mm256_loadu_ps(&lastLayerActivations[j]);
                acc0 = _mm256_fmadd_ps(w, a, acc0);
            }

            __m128 lo128  = _mm256_castps256_ps128(acc0);
            __m128 hi128  = _mm256_extractf128_ps(acc0, 1);
            __m128 sum128 = _mm_add_ps(lo128, hi128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            data_t sum = _mm_cvtss_f32(sum128);

            for (; j < cols; j++)
                sum += weights[row_offset + j] * lastLayerActivations[j];

            v_out_loc[i] = sum;
        }
        return 1;
    }

    return 0;
}

int kernel_vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out) {

    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutrow = get_matrix_rows(v_out);
    int voutcol = get_matrix_cols(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_matrix_start(v_out);

    if (v1len == voutrow && v2len == voutcol) {
        for (int i = 0; i < v1len; i++) {
            int    row_offset = i * voutcol;
            data_t v1_val     = v1_start[i];

            __m256 v1_vec = _mm256_set1_ps(v1_val);

            int j;

            for (j = 0; j < v2len - (AVX_STRIDE_6 - 1); j += AVX_STRIDE_6) {
                __m256 v2_0 = _mm256_loadu_ps(&v2_start[j]);
                _mm256_storeu_ps(&vout_start[row_offset + j],
                                 _mm256_mul_ps(v1_vec, v2_0));

                __m256 v2_1 = _mm256_loadu_ps(&v2_start[j + AVX_STRIDE]);
                _mm256_storeu_ps(&vout_start[row_offset + j + AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, v2_1));

                __m256 v2_2 = _mm256_loadu_ps(&v2_start[j + 2*AVX_STRIDE]);
                _mm256_storeu_ps(&vout_start[row_offset + j + 2*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, v2_2));

                __m256 v2_3 = _mm256_loadu_ps(&v2_start[j + 3*AVX_STRIDE]);
                _mm256_storeu_ps(&vout_start[row_offset + j + 3*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, v2_3));

                __m256 v2_4 = _mm256_loadu_ps(&v2_start[j + 4*AVX_STRIDE]);
                _mm256_storeu_ps(&vout_start[row_offset + j + 4*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, v2_4));

                __m256 v2_5 = _mm256_loadu_ps(&v2_start[j + 5*AVX_STRIDE]);
                _mm256_storeu_ps(&vout_start[row_offset + j + 5*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, v2_5));
            }

            for (; j < v2len - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 v2_vec = _mm256_loadu_ps(&v2_start[j]);
                _mm256_storeu_ps(&vout_start[row_offset + j],
                                 _mm256_mul_ps(v1_vec, v2_vec));
            }

            for (; j < v2len; j++)
                vout_start[row_offset + j] = v1_val * v2_start[j];
        }
        return 1;
    }

    return 0;
}

int kernel_matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out) {

    long int rows1    = get_matrix_rows(m1);
    long int cols1    = get_matrix_cols(m1);
    long int rows2    = get_matrix_rows(m2);
    long int cols2    = get_matrix_cols(m2);
    long int rows_out = get_matrix_rows(m_out);
    long int cols_out = get_matrix_cols(m_out);

    data_t* restrict m1_start    = get_matrix_start(m1);
    data_t* restrict m2_start    = get_matrix_start(m2);
    data_t* restrict m_out_start = get_matrix_start(m_out);

    if (rows1 == rows2 && cols1 == cols2 && rows1 == rows_out && cols1 == cols_out) {
        for (int i = 0; i < rows1; i++) {
            long int row_offset = (long int)i * cols1;

            int j;

            for (j = 0; j < cols1 - (AVX_STRIDE_6 - 1); j += AVX_STRIDE_6) {
                __m256 a0 = _mm256_loadu_ps(&m1_start[row_offset + j]);
                __m256 b0 = _mm256_loadu_ps(&m2_start[row_offset + j]);
                _mm256_storeu_ps(&m_out_start[row_offset + j], _mm256_add_ps(a0, b0));

                __m256 a1 = _mm256_loadu_ps(&m1_start[row_offset + j + AVX_STRIDE]);
                __m256 b1 = _mm256_loadu_ps(&m2_start[row_offset + j + AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + AVX_STRIDE], _mm256_add_ps(a1, b1));

                __m256 a2 = _mm256_loadu_ps(&m1_start[row_offset + j + 2*AVX_STRIDE]);
                __m256 b2 = _mm256_loadu_ps(&m2_start[row_offset + j + 2*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 2*AVX_STRIDE], _mm256_add_ps(a2, b2));

                __m256 a3 = _mm256_loadu_ps(&m1_start[row_offset + j + 3*AVX_STRIDE]);
                __m256 b3 = _mm256_loadu_ps(&m2_start[row_offset + j + 3*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 3*AVX_STRIDE], _mm256_add_ps(a3, b3));

                __m256 a4 = _mm256_loadu_ps(&m1_start[row_offset + j + 4*AVX_STRIDE]);
                __m256 b4 = _mm256_loadu_ps(&m2_start[row_offset + j + 4*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 4*AVX_STRIDE], _mm256_add_ps(a4, b4));

                __m256 a5 = _mm256_loadu_ps(&m1_start[row_offset + j + 5*AVX_STRIDE]);
                __m256 b5 = _mm256_loadu_ps(&m2_start[row_offset + j + 5*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 5*AVX_STRIDE], _mm256_add_ps(a5, b5));
            }

            for (; j < cols1 - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 a = _mm256_loadu_ps(&m1_start[row_offset + j]);
                __m256 b = _mm256_loadu_ps(&m2_start[row_offset + j]);
                _mm256_storeu_ps(&m_out_start[row_offset + j], _mm256_add_ps(a, b));
            }

            for (; j < cols1; j++)
                m_out_start[row_offset + j] = m1_start[row_offset + j] + m2_start[row_offset + j];
        }
        return 1;
    }

    return 0;
}

int kernel_matrix_saxpy(matrix_ptr grad, data_t scale, matrix_ptr W) {

    long int rows  = get_matrix_rows(W);
    long int cols  = get_matrix_cols(W);
    long int grows = get_matrix_rows(grad);
    long int gcols = get_matrix_cols(grad);

    if (rows != grows || cols != gcols) return 0;

    long int n = rows * cols;

    data_t* restrict w_data    = get_matrix_start(W);
    data_t* restrict grad_data = get_matrix_start(grad);

    __m256 scale_vec = _mm256_set1_ps(scale);

    long int i;

    for (i = 0; i < n - 47; i += 48) {
        __m256 w0 = _mm256_loadu_ps(&w_data[i]);
        __m256 g0 = _mm256_loadu_ps(&grad_data[i]);
        _mm256_storeu_ps(&w_data[i], _mm256_fmadd_ps(scale_vec, g0, w0));

        __m256 w1 = _mm256_loadu_ps(&w_data[i + 8]);
        __m256 g1 = _mm256_loadu_ps(&grad_data[i + 8]);
        _mm256_storeu_ps(&w_data[i + 8], _mm256_fmadd_ps(scale_vec, g1, w1));

        __m256 w2 = _mm256_loadu_ps(&w_data[i + 16]);
        __m256 g2 = _mm256_loadu_ps(&grad_data[i + 16]);
        _mm256_storeu_ps(&w_data[i + 16], _mm256_fmadd_ps(scale_vec, g2, w2));

        __m256 w3 = _mm256_loadu_ps(&w_data[i + 24]);
        __m256 g3 = _mm256_loadu_ps(&grad_data[i + 24]);
        _mm256_storeu_ps(&w_data[i + 24], _mm256_fmadd_ps(scale_vec, g3, w3));

        __m256 w4 = _mm256_loadu_ps(&w_data[i + 32]);
        __m256 g4 = _mm256_loadu_ps(&grad_data[i + 32]);
        _mm256_storeu_ps(&w_data[i + 32], _mm256_fmadd_ps(scale_vec, g4, w4));

        __m256 w5 = _mm256_loadu_ps(&w_data[i + 40]);
        __m256 g5 = _mm256_loadu_ps(&grad_data[i + 40]);
        _mm256_storeu_ps(&w_data[i + 40], _mm256_fmadd_ps(scale_vec, g5, w5));
    }
    for (; i < n - 7; i += 8) {
        __m256 w = _mm256_loadu_ps(&w_data[i]);
        __m256 g = _mm256_loadu_ps(&grad_data[i]);
        _mm256_storeu_ps(&w_data[i], _mm256_fmadd_ps(scale_vec, g, w));
    }
    for (; i < n; i++)
        w_data[i] += scale * grad_data[i];

    return 1;
}

int kernel_vector_saxpy(array_ptr grad, data_t scale, array_ptr b) {

    long int n  = b->len;
    long int gn = grad->len;

    if (n != gn) return 0;

    data_t* restrict b_data    = b->data;
    data_t* restrict grad_data = grad->data;

    __m256 scale_vec = _mm256_set1_ps(scale);

    long int i;

    for (i = 0; i < n - 47; i += 48) {
        __m256 b0 = _mm256_loadu_ps(&b_data[i]);
        __m256 g0 = _mm256_loadu_ps(&grad_data[i]);
        _mm256_storeu_ps(&b_data[i], _mm256_fmadd_ps(scale_vec, g0, b0));

        __m256 b1 = _mm256_loadu_ps(&b_data[i + 8]);
        __m256 g1 = _mm256_loadu_ps(&grad_data[i + 8]);
        _mm256_storeu_ps(&b_data[i + 8], _mm256_fmadd_ps(scale_vec, g1, b1));

        __m256 b2 = _mm256_loadu_ps(&b_data[i + 16]);
        __m256 g2 = _mm256_loadu_ps(&grad_data[i + 16]);
        _mm256_storeu_ps(&b_data[i + 16], _mm256_fmadd_ps(scale_vec, g2, b2));

        __m256 b3 = _mm256_loadu_ps(&b_data[i + 24]);
        __m256 g3 = _mm256_loadu_ps(&grad_data[i + 24]);
        _mm256_storeu_ps(&b_data[i + 24], _mm256_fmadd_ps(scale_vec, g3, b3));

        __m256 b4 = _mm256_loadu_ps(&b_data[i + 32]);
        __m256 g4 = _mm256_loadu_ps(&grad_data[i + 32]);
        _mm256_storeu_ps(&b_data[i + 32], _mm256_fmadd_ps(scale_vec, g4, b4));

        __m256 b5 = _mm256_loadu_ps(&b_data[i + 40]);
        __m256 g5 = _mm256_loadu_ps(&grad_data[i + 40]);
        _mm256_storeu_ps(&b_data[i + 40], _mm256_fmadd_ps(scale_vec, g5, b5));
    }
    for (; i < n - 7; i += 8) {
        __m256 bv = _mm256_loadu_ps(&b_data[i]);
        __m256 gv = _mm256_loadu_ps(&grad_data[i]);
        _mm256_storeu_ps(&b_data[i], _mm256_fmadd_ps(scale_vec, gv, bv));
    }
    for (; i < n; i++)
        b_data[i] += scale * grad_data[i];

    return 1;
}

int kernel_vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out) {

    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutlen = get_array_length(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_array_start(v_out);

    if (v1len == v2len && v2len == voutlen) {
        int len = v1len;
        int i;

        for (i = 0; i < len - (AVX_STRIDE - 1); i += AVX_STRIDE) {
            __m256 a = _mm256_loadu_ps(&v1_start[i]);
            __m256 b = _mm256_loadu_ps(&v2_start[i]);
            _mm256_storeu_ps(&vout_start[i], _mm256_add_ps(a, b));
        }

        for (; i < len; i++)
            vout_start[i] = v1_start[i] + v2_start[i];

        return 1;
    }

    return 0;
}

int kernel_matrix_transpose(matrix_ptr m, matrix_ptr m_out) {

    long int rows     = get_matrix_rows(m);
    long int cols     = get_matrix_cols(m);
    long int out_rows = get_matrix_rows(m_out);
    long int out_cols = get_matrix_cols(m_out);

    data_t* restrict m_start     = get_matrix_start(m);
    data_t* restrict m_out_start = get_matrix_start(m_out);

    if (rows == out_cols && cols == out_rows) {
        long int s1 = out_cols;
        long int s2 = 2L * out_cols;
        long int s3 = 3L * out_cols;
        long int s4 = 4L * out_cols;
        long int s5 = 5L * out_cols;
        long int s6 = 6L * out_cols;

        int i, j;

        for (i = 0; i < rows; i++) {
            long int row_offset = (long int)i * cols;
            long int out_idx    = (long int)i;

            for (j = 0; j < cols - 5; j += 6, out_idx += s6) {
                m_out_start[out_idx]      = m_start[row_offset + j];
                m_out_start[out_idx + s1] = m_start[row_offset + j + 1];
                m_out_start[out_idx + s2] = m_start[row_offset + j + 2];
                m_out_start[out_idx + s3] = m_start[row_offset + j + 3];
                m_out_start[out_idx + s4] = m_start[row_offset + j + 4];
                m_out_start[out_idx + s5] = m_start[row_offset + j + 5];
            }

            for (; j < cols; j++, out_idx += out_cols)
                m_out_start[out_idx] = m_start[row_offset + j];
        }
        return 1;
    }

    return 0;
}

int kernel_matrix_scalar_mult(matrix_ptr m, data_t scalar, matrix_ptr m_out) {
    long int rows     = get_matrix_rows(m);
    long int cols     = get_matrix_cols(m);
    long int out_rows = get_matrix_rows(m_out);
    long int out_cols = get_matrix_cols(m_out);

    data_t* m_start     = get_matrix_start(m);
    data_t* m_out_start = get_matrix_start(m_out);

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    if (rows == out_rows && cols == out_cols) {
        for (int i = 0; i < rows; i++) {
            int row_offset = i * cols;
            int j;
            for (j = 0; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 in  = _mm256_loadu_ps(&m_start[row_offset + j]);
                _mm256_storeu_ps(&m_out_start[row_offset + j], _mm256_mul_ps(scalar_vec, in));
            }
            for (; j < cols; j++)
                m_out_start[row_offset + j] = scalar * m_start[row_offset + j];
        }
        return 1;
    }
    return 0;
}

int kernel_vector_scalar_mult(array_ptr v1, data_t scalar, array_ptr v_out) {
    int len     = get_array_length(v1);
    int out_len = get_array_length(v_out);

    data_t* v1_start    = get_array_start(v1);
    data_t* v_out_start = get_array_start(v_out);

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    if (len == out_len) {
        int j;
        for (j = 0; j < len - (AVX_STRIDE - 1); j += AVX_STRIDE) {
            __m256 in  = _mm256_loadu_ps(&v1_start[j]);
            _mm256_storeu_ps(&v_out_start[j], _mm256_mul_ps(scalar_vec, in));
        }
        for (; j < len; j++)
            v_out_start[j] = scalar * v1_start[j];
        return 1;
    }
    return 0;
}

void kernel_sigmoid_arr(array_ptr v) {
    int len = get_array_length(v);
    data_t* v_start = get_array_start(v);

    __m256 pointfive = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);

    int i;
    for (i = 0; i < len - (AVX_STRIDE - 1); i += AVX_STRIDE) {
        __m256 in      = _mm256_loadu_ps(&v_start[i]);
        __m256 intermed = _mm256_andnot_ps(sign_mask, in);
        intermed = _mm256_add_ps(one, intermed);
        intermed = _mm256_div_ps(in, intermed);
        intermed = _mm256_mul_ps(pointfive, intermed);
        intermed = _mm256_add_ps(pointfive, intermed);
        _mm256_storeu_ps(&v_start[i], intermed);
    }
    for (; i < len; i++) {
        float x = v_start[i];
        v_start[i] = 0.5f * (x / (1.0f + (x < 0.0f ? -x : x))) + 0.5f;
    }
}


// =====================================================================
// New batch kernels
// =====================================================================

/* kernel_gemm_forward — C = A x B^T  (dot-product / reduction form)
 *
 * A: (actual_S x k)   — batch of activations, samples as rows
 * B: (m x k)          — weight matrix (or transposed weight for error prop)
 * C: (actual_S x m)   — output batch, samples as rows
 *
 * C[s][o] = sum_{j=0}^{k-1} A[s][j] * B[o][j]
 *
 * Inner loop is over k (the feature / weight dimension).
 * For the first layer: k=784, giving 98 full AVX passes per (s,o) pair.
 * AVX+FMA, no unrolling, no tiling.
 */
int kernel_gemm_forward(matrix_ptr A, matrix_ptr B, matrix_ptr C, int actual_S) {

    long int k = get_matrix_cols(A);   /* inner (feature) dimension */
    long int m = get_matrix_rows(B);   /* number of output neurons  */

    if (get_matrix_cols(B) != k) return 0;

    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    for (int s = 0; s < actual_S; s++) {
        long int a_row = (long int)s * k;
        long int c_row = (long int)s * m;

        for (long int o = 0; o < m; o++) {
            long int b_row = o * k;

            __m256 acc = _mm256_setzero_ps();

            long int j;
            for (j = 0; j < k - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 av = _mm256_loadu_ps(&a[a_row + j]);
                __m256 bv = _mm256_loadu_ps(&b[b_row + j]);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }

            /* Horizontal reduction: fold 8 lanes to 1 scalar */
            __m128 lo  = _mm256_castps256_ps128(acc);
            __m128 hi  = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            data_t val = _mm_cvtss_f32(sum);

            /* Scalar tail: 0-7 remaining elements */
            for (; j < k; j++)
                val += a[a_row + j] * b[b_row + j];

            c[c_row + o] = val;
        }
    }
    return 1;
}

/* kernel_gemm_weight_grad — dW += delta^T x act  (accumulating saxpy form)
 *
 * delta: (actual_S x out_dim)  — batch of deltas, samples as rows
 * act:   (actual_S x in_dim)   — batch of activations, samples as rows
 * dW:    (out_dim x in_dim)    — weight gradient matrix (accumulated into)
 *
 * dW[o][j] += sum_{s=0}^{actual_S-1} delta[s][o] * act[s][j]
 *
 * Inner loop is over in_dim (j).  For the first layer: in_dim=784.
 * delta[s][o] is a scalar broadcast; dW and act are stride-1 in j.
 * AVX+FMA, no unrolling, no tiling.
 */
int kernel_gemm_weight_grad(matrix_ptr delta, matrix_ptr act, matrix_ptr dW, int actual_S) {

    long int out_dim = get_matrix_cols(delta);
    long int in_dim  = get_matrix_cols(act);

    if (get_matrix_rows(dW) != out_dim || get_matrix_cols(dW) != in_dim) return 0;

    data_t* restrict d  = get_matrix_start(delta);
    data_t* restrict a  = get_matrix_start(act);
    data_t* restrict gw = get_matrix_start(dW);

    for (long int o = 0; o < out_dim; o++) {
        long int dw_row = o * in_dim;

        for (int s = 0; s < actual_S; s++) {
            data_t d_so  = d[(long int)s * out_dim + o];
            __m256 d_vec = _mm256_set1_ps(d_so);

            long int a_row = (long int)s * in_dim;

            long int j;
            for (j = 0; j < in_dim - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 wv = _mm256_loadu_ps(&gw[dw_row + j]);
                __m256 av = _mm256_loadu_ps(&a[a_row + j]);
                _mm256_storeu_ps(&gw[dw_row + j], _mm256_fmadd_ps(d_vec, av, wv));
            }

            /* Scalar tail */
            for (; j < in_dim; j++)
                gw[dw_row + j] += d_so * a[a_row + j];
        }
    }
    return 1;
}

/* kernel_sigmoid_mat — apply AVX approximate sigmoid in-place to first
 * actual_S rows of m.  Same approximation as kernel_sigmoid_arr.
 * f(x) = 0.5 * x/(1+|x|) + 0.5
 */
void kernel_sigmoid_mat(matrix_ptr m, int actual_S) {
    long int cols  = get_matrix_cols(m);
    long int total = (long int)actual_S * cols;
    data_t*  v     = get_matrix_start(m);

    __m256 half      = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);

    long int i;
    for (i = 0; i < total - (AVX_STRIDE - 1); i += AVX_STRIDE) {
        __m256 x   = _mm256_loadu_ps(&v[i]);
        __m256 ax  = _mm256_andnot_ps(sign_mask, x);
        __m256 den = _mm256_add_ps(one, ax);
        __m256 out = _mm256_fmadd_ps(half, _mm256_div_ps(x, den), half);
        _mm256_storeu_ps(&v[i], out);
    }
    for (; i < total; i++) {
        float x = v[i];
        v[i] = 0.5f * (x / (1.0f + (x < 0.0f ? -x : x))) + 0.5f;
    }
}

/* kernel_sigmoid_prime_mat — apply sigmoid' in-place to first actual_S rows.
 * sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
 * Applied to the stored pre-activation z values.
 */
void kernel_sigmoid_prime_mat(matrix_ptr m, int actual_S) {
    long int cols  = get_matrix_cols(m);
    long int total = (long int)actual_S * cols;
    data_t*  v     = get_matrix_start(m);

    __m256 half      = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);

    long int i;
    for (i = 0; i < total - (AVX_STRIDE - 1); i += AVX_STRIDE) {
        /* compute sig(z) using the same approximation */
        __m256 z   = _mm256_loadu_ps(&v[i]);
        __m256 az  = _mm256_andnot_ps(sign_mask, z);
        __m256 den = _mm256_add_ps(one, az);
        __m256 sg  = _mm256_fmadd_ps(half, _mm256_div_ps(z, den), half);
        /* sig * (1 - sig) */
        __m256 sp  = _mm256_mul_ps(sg, _mm256_sub_ps(one, sg));
        _mm256_storeu_ps(&v[i], sp);
    }
    for (; i < total; i++) {
        float z  = v[i];
        float sg = 0.5f * (z / (1.0f + (z < 0.0f ? -z : z))) + 0.5f;
        v[i] = sg * (1.0f - sg);
    }
}

/* kernel_hadamard_mat — element-wise multiply: C[s][j] = A[s][j] * B[s][j]
 * Operates on the first actual_S rows.  A, B, C must have the same shape.
 */
int kernel_hadamard_mat(matrix_ptr A, matrix_ptr B, matrix_ptr C, int actual_S) {

    long int cols  = get_matrix_cols(A);
    long int total = (long int)actual_S * cols;

    if (get_matrix_cols(B) != cols || get_matrix_cols(C) != cols) return 0;

    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    long int i;
    for (i = 0; i < total - (AVX_STRIDE - 1); i += AVX_STRIDE) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(av, bv));
    }
    for (; i < total; i++)
        c[i] = a[i] * b[i];

    return 1;
}

/* kernel_bias_broadcast_add — Z[s][j] += b[j]  for all s in 0..actual_S-1.
 * Z has shape (max_S x cols); b has length cols.
 * The inner j-loop is vectorized; it is stride-1 on both Z and b.
 */
int kernel_bias_broadcast_add(matrix_ptr Z, array_ptr b, int actual_S) {

    long int cols = get_matrix_cols(Z);

    if (get_array_length(b) != cols) return 0;

    data_t* restrict z_data = get_matrix_start(Z);
    data_t* restrict b_data = get_array_start(b);

    for (int s = 0; s < actual_S; s++) {
        long int row = (long int)s * cols;
        long int j;
        for (j = 0; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
            __m256 zv = _mm256_loadu_ps(&z_data[row + j]);
            __m256 bv = _mm256_loadu_ps(&b_data[j]);
            _mm256_storeu_ps(&z_data[row + j], _mm256_add_ps(zv, bv));
        }
        for (; j < cols; j++)
            z_data[row + j] += b_data[j];
    }
    return 1;
}

/* kernel_bias_grad_accum — b_sum[j] += sum_{s=0}^{actual_S-1} delta[s][j]
 * delta has shape (max_S x cols); b_sum has length cols.
 * Column-sum of the first actual_S rows, accumulated into b_sum.
 * The inner j-loop is vectorized (stride-1 on both delta row and b_sum).
 */
int kernel_bias_grad_accum(matrix_ptr delta, array_ptr b_sum, int actual_S) {

    long int cols = get_matrix_cols(delta);

    if (get_array_length(b_sum) != cols) return 0;

    data_t* restrict d    = get_matrix_start(delta);
    data_t* restrict bsum = get_array_start(b_sum);

    for (int s = 0; s < actual_S; s++) {
        long int row = (long int)s * cols;
        long int j;
        for (j = 0; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
            __m256 dv  = _mm256_loadu_ps(&d[row + j]);
            __m256 bv  = _mm256_loadu_ps(&bsum[j]);
            _mm256_storeu_ps(&bsum[j], _mm256_add_ps(bv, dv));
        }
        for (; j < cols; j++)
            bsum[j] += d[row + j];
    }
    return 1;
}

/* kernel_matrix_copy_rows — copy the first actual_S rows from src to dst.
 * src and dst must have the same number of columns.
 */
int kernel_matrix_copy_rows(matrix_ptr src, matrix_ptr dst, int actual_S) {

    long int cols = get_matrix_cols(src);
    if (get_matrix_cols(dst) != cols) return 0;

    memcpy(get_matrix_start(dst),
           get_matrix_start(src),
           (size_t)actual_S * cols * sizeof(data_t));
    return 1;
}
