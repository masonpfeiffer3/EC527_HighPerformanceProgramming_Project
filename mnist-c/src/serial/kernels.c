#include "kernels.h"
#include "array_matrix_funcs.h"
#include <immintrin.h>          /* AVX intrinsics */

/* 256-bit AVX register holds 8 x float32 */
#define AVX_STRIDE 8

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

            /* 8-lane accumulator, all zeros: {0, 0, 0, 0, 0, 0, 0, 0} */
            __m256 acc = _mm256_setzero_ps();

            int j;
            /* --- AVX loop: process 8 elements per iteration ---
             * Bound (cols - AVX_STRIDE + 1) ensures indices j to j+7
             * are always valid; for cols=784 (divisible by 8) this
             * covers all elements with no remainder.                  */
            for (j = 0; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {

                /* Load 8 weights from this row: weights[row_offset+j .. +j+7] */
                __m256 w_vec = _mm256_loadu_ps(&weights[row_offset + j]);

                /* Load 8 activations: lastLayerActivations[j .. j+7] */
                __m256 a_vec = _mm256_loadu_ps(&lastLayerActivations[j]);

                /* Element-wise multiply: lane k = w[j+k] * a[j+k] */
                __m256 prod  = _mm256_mul_ps(w_vec, a_vec);

                /* Accumulate into 8-lane running sum */
                acc = _mm256_add_ps(acc, prod);
            }

            /* --- Horizontal reduction ---
             * acc[k] holds the sum of products at positions k, k+8, k+16...
             * We must sum all 8 lanes into one scalar.
             * Strategy: store to a temp array and sum scalarly.
             * (A shuffle-based reduction would be faster, but is left
             *  for a later optimization step.)                         */
            float temp[AVX_STRIDE];
            _mm256_storeu_ps(temp, acc);
            data_t sum = temp[0] + temp[1] + temp[2] + temp[3]
                       + temp[4] + temp[5] + temp[6] + temp[7];

            /* --- Scalar cleanup: handles remaining elements (up to 7) ---
             * For cols=784 (divisible by 8), this loop body never executes. */
            for (; j < cols; j++) {
                sum += weights[row_offset + j] * lastLayerActivations[j];
            }

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
            int row_offset = i * voutcol;
            data_t v1_val     = v1_start[i];

            /* Broadcast scalar v1[i] to all 8 lanes:
             * v1_vec = {v1i, v1i, v1i, v1i, v1i, v1i, v1i, v1i}     */
            __m256 v1_vec = _mm256_set1_ps(v1_val);

            int j;
            /* --- AVX loop: multiply broadcast v1[i] by 8 v2 elements ---
             * No reduction needed; store 8 outputs per iteration.     */
            for (j = 0; j < v2len - (AVX_STRIDE - 1); j += AVX_STRIDE) {

                /* Load 8 elements of v2 */
                __m256 v2_vec = _mm256_loadu_ps(&v2_start[j]);

                /* Element-wise multiply: lane k = v1[i] * v2[j+k] */
                __m256 result = _mm256_mul_ps(v1_vec, v2_vec);

                /* Store 8 products directly into output matrix row i */
                _mm256_storeu_ps(&vout_start[row_offset + j], result);
            }

            /* Scalar cleanup for remaining elements (up to 7) */
            for (; j < v2len; j++) {
                vout_start[row_offset + j] = v1_val * v2_start[j];
            }
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
            /* --- AVX loop: add 8 element pairs per iteration ---
             * No reduction or broadcast needed; pure element-wise add. */
            for (j = 0; j < cols1 - (AVX_STRIDE - 1); j += AVX_STRIDE) {

                /* Load 8 elements from each input matrix */
                __m256 a = _mm256_loadu_ps(&m1_start[row_offset + j]);
                __m256 b = _mm256_loadu_ps(&m2_start[row_offset + j]);

                /* Element-wise add: lane k = m1[row+j+k] + m2[row+j+k] */
                __m256 c = _mm256_add_ps(a, b);

                /* Store 8 results to output matrix */
                _mm256_storeu_ps(&m_out_start[row_offset + j], c);
            }

            /* Scalar cleanup for remaining elements (up to 7) */
            for (; j < cols1; j++) {
                m_out_start[row_offset + j] =
                    m1_start[row_offset + j] + m2_start[row_offset + j];
            }
        }
        return 1;
    }

    return 0;
}