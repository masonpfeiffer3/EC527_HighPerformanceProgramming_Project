#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>           /* AVX, AVX2, FMA */

#include "array_matrix_math.h"
#include "array_matrix_funcs.h"

#define AVX_STRIDE   8
#define AVX_STRIDE_6 48          /* 6 * AVX_STRIDE */


/* ===========================================================================
 * matrix_transpose -- 6x scalar unroll with hoisted strides.
 * Bandwidth-bound; AVX shuffles for transpose are complex, so we just
 * unroll to amortize loop overhead.
 * =========================================================================== */
int matrix_transpose(matrix_ptr m, matrix_ptr m_out) {
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

        for (int i = 0; i < rows; i++) {
            long int row_offset = (long int)i * cols;
            long int out_idx    = (long int)i;

            int j;
            for (j = 0; j < cols - 5; j += 6, out_idx += s6) {
                m_out_start[out_idx     ] = m_start[row_offset + j    ];
                m_out_start[out_idx + s1] = m_start[row_offset + j + 1];
                m_out_start[out_idx + s2] = m_start[row_offset + j + 2];
                m_out_start[out_idx + s3] = m_start[row_offset + j + 3];
                m_out_start[out_idx + s4] = m_start[row_offset + j + 4];
                m_out_start[out_idx + s5] = m_start[row_offset + j + 5];
            }
            for (; j < cols; j++, out_idx += out_cols) {
                m_out_start[out_idx] = m_start[row_offset + j];
            }
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * matrix_scalar_mult -- AVX broadcast scalar, vectorize inner cols loop.
 * Used per-batch for parameter update scaling (3 scaling ops per layer).
 * =========================================================================== */
int matrix_scalar_mult(matrix_ptr m, data_t scalar, matrix_ptr m_out) {
    long int rows     = get_matrix_rows(m);
    long int cols     = get_matrix_cols(m);
    long int out_rows = get_matrix_rows(m_out);
    long int out_cols = get_matrix_cols(m_out);

    data_t* restrict m_start     = get_matrix_start(m);
    data_t* restrict m_out_start = get_matrix_start(m_out);

    if (rows == out_rows && cols == out_cols) {
        __m256 sv = _mm256_set1_ps(scalar);

        for (int i = 0; i < rows; i++) {
            long int row_off = (long int)i * cols;
            int j;
            for (j = 0; j <= cols - AVX_STRIDE; j += AVX_STRIDE) {
                __m256 in = _mm256_loadu_ps(&m_start[row_off + j]);
                _mm256_storeu_ps(&m_out_start[row_off + j], _mm256_mul_ps(sv, in));
            }
            for (; j < cols; j++) {
                m_out_start[row_off + j] = scalar * m_start[row_off + j];
            }
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * matrix_vector_mult -- 6x unrolled FMA with 6 independent accumulators.
 * Used in test_parallel_MNIST() for single-sample inference.
 * =========================================================================== */
int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out) {
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
            for (j = 0; j <= cols - AVX_STRIDE_6; j += AVX_STRIDE_6) {
                __m256 w0 = _mm256_loadu_ps(&weights[row_offset + j              ]);
                __m256 w1 = _mm256_loadu_ps(&weights[row_offset + j +   AVX_STRIDE]);
                __m256 w2 = _mm256_loadu_ps(&weights[row_offset + j + 2*AVX_STRIDE]);
                __m256 w3 = _mm256_loadu_ps(&weights[row_offset + j + 3*AVX_STRIDE]);
                __m256 w4 = _mm256_loadu_ps(&weights[row_offset + j + 4*AVX_STRIDE]);
                __m256 w5 = _mm256_loadu_ps(&weights[row_offset + j + 5*AVX_STRIDE]);

                __m256 a0 = _mm256_loadu_ps(&lastLayerActivations[j              ]);
                __m256 a1 = _mm256_loadu_ps(&lastLayerActivations[j +   AVX_STRIDE]);
                __m256 a2 = _mm256_loadu_ps(&lastLayerActivations[j + 2*AVX_STRIDE]);
                __m256 a3 = _mm256_loadu_ps(&lastLayerActivations[j + 3*AVX_STRIDE]);
                __m256 a4 = _mm256_loadu_ps(&lastLayerActivations[j + 4*AVX_STRIDE]);
                __m256 a5 = _mm256_loadu_ps(&lastLayerActivations[j + 5*AVX_STRIDE]);

                acc0 = _mm256_fmadd_ps(w0, a0, acc0);
                acc1 = _mm256_fmadd_ps(w1, a1, acc1);
                acc2 = _mm256_fmadd_ps(w2, a2, acc2);
                acc3 = _mm256_fmadd_ps(w3, a3, acc3);
                acc4 = _mm256_fmadd_ps(w4, a4, acc4);
                acc5 = _mm256_fmadd_ps(w5, a5, acc5);
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc0 = _mm256_add_ps(acc0, acc4);

            for (; j <= cols - AVX_STRIDE; j += AVX_STRIDE) {
                __m256 w = _mm256_loadu_ps(&weights[row_offset + j]);
                __m256 a = _mm256_loadu_ps(&lastLayerActivations[j]);
                acc0 = _mm256_fmadd_ps(w, a, acc0);
            }

            __m128 lo  = _mm256_castps256_ps128(acc0);
            __m128 hi  = _mm256_extractf128_ps(acc0, 1);
            __m128 s4  = _mm_add_ps(lo, hi);
            s4 = _mm_hadd_ps(s4, s4);
            s4 = _mm_hadd_ps(s4, s4);
            data_t sum = _mm_cvtss_f32(s4);

            for (; j < cols; j++) {
                sum += weights[row_offset + j] * lastLayerActivations[j];
            }

            v_out_loc[i] = sum;
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * matrix_matrix_add -- 6x unrolled AVX. Used per-batch for:
 *   1. Gradient reduction: H0_W_grad_sum += per-thread gradient
 *   2. Parameter update:   H0_W += scaled_gradient
 * =========================================================================== */
int matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out) {
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
            long int row_off = (long int)i * cols1;

            int j;
            for (j = 0; j <= cols1 - AVX_STRIDE_6; j += AVX_STRIDE_6) {
                __m256 a0 = _mm256_loadu_ps(&m1_start[row_off + j              ]);
                __m256 b0 = _mm256_loadu_ps(&m2_start[row_off + j              ]);
                _mm256_storeu_ps(&m_out_start[row_off + j], _mm256_add_ps(a0, b0));

                __m256 a1 = _mm256_loadu_ps(&m1_start[row_off + j +   AVX_STRIDE]);
                __m256 b1 = _mm256_loadu_ps(&m2_start[row_off + j +   AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_off + j + AVX_STRIDE], _mm256_add_ps(a1, b1));

                __m256 a2 = _mm256_loadu_ps(&m1_start[row_off + j + 2*AVX_STRIDE]);
                __m256 b2 = _mm256_loadu_ps(&m2_start[row_off + j + 2*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_off + j + 2*AVX_STRIDE], _mm256_add_ps(a2, b2));

                __m256 a3 = _mm256_loadu_ps(&m1_start[row_off + j + 3*AVX_STRIDE]);
                __m256 b3 = _mm256_loadu_ps(&m2_start[row_off + j + 3*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_off + j + 3*AVX_STRIDE], _mm256_add_ps(a3, b3));

                __m256 a4 = _mm256_loadu_ps(&m1_start[row_off + j + 4*AVX_STRIDE]);
                __m256 b4 = _mm256_loadu_ps(&m2_start[row_off + j + 4*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_off + j + 4*AVX_STRIDE], _mm256_add_ps(a4, b4));

                __m256 a5 = _mm256_loadu_ps(&m1_start[row_off + j + 5*AVX_STRIDE]);
                __m256 b5 = _mm256_loadu_ps(&m2_start[row_off + j + 5*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_off + j + 5*AVX_STRIDE], _mm256_add_ps(a5, b5));
            }

            for (; j <= cols1 - AVX_STRIDE; j += AVX_STRIDE) {
                __m256 a = _mm256_loadu_ps(&m1_start[row_off + j]);
                __m256 b = _mm256_loadu_ps(&m2_start[row_off + j]);
                _mm256_storeu_ps(&m_out_start[row_off + j], _mm256_add_ps(a, b));
            }

            for (; j < cols1; j++) {
                m_out_start[row_off + j] = m1_start[row_off + j] + m2_start[row_off + j];
            }
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_vector_add -- AVX flat. Bias gradient reduction & parameter update.
 * =========================================================================== */
int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out) {
    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutlen = get_array_length(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_array_start(v_out);

    if (v1len == v2len && v2len == voutlen) {
        int len = v1len;
        int i;
        for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
            __m256 a = _mm256_loadu_ps(&v1_start[i]);
            __m256 b = _mm256_loadu_ps(&v2_start[i]);
            _mm256_storeu_ps(&vout_start[i], _mm256_add_ps(a, b));
        }
        for (; i < len; i++) {
            vout_start[i] = v1_start[i] + v2_start[i];
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_vector_sub -- AVX flat.
 * =========================================================================== */
int vector_vector_sub(array_ptr v1, array_ptr v2, array_ptr v_out) {
    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutlen = get_array_length(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_array_start(v_out);

    if (v1len == v2len && v2len == voutlen) {
        int len = v1len;
        int i;
        for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
            __m256 a = _mm256_loadu_ps(&v1_start[i]);
            __m256 b = _mm256_loadu_ps(&v2_start[i]);
            _mm256_storeu_ps(&vout_start[i], _mm256_sub_ps(a, b));
        }
        for (; i < len; i++) {
            vout_start[i] = v1_start[i] - v2_start[i];
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_vector_mult -- outer product. 6x unrolled with broadcast scalar.
 * =========================================================================== */
int vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out) {
    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutrow = get_matrix_rows(v_out);
    int voutcol = get_matrix_cols(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_matrix_start(v_out);

    if (v1len == voutrow && v2len == voutcol) {
        for (int i = 0; i < v1len; i++) {
            long int row_offset = (long int)i * voutcol;
            __m256 v1_vec = _mm256_set1_ps(v1_start[i]);

            int j;
            for (j = 0; j <= v2len - AVX_STRIDE_6; j += AVX_STRIDE_6) {
                _mm256_storeu_ps(&vout_start[row_offset + j              ],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j              ])));
                _mm256_storeu_ps(&vout_start[row_offset + j +   AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j +   AVX_STRIDE])));
                _mm256_storeu_ps(&vout_start[row_offset + j + 2*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j + 2*AVX_STRIDE])));
                _mm256_storeu_ps(&vout_start[row_offset + j + 3*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j + 3*AVX_STRIDE])));
                _mm256_storeu_ps(&vout_start[row_offset + j + 4*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j + 4*AVX_STRIDE])));
                _mm256_storeu_ps(&vout_start[row_offset + j + 5*AVX_STRIDE],
                                 _mm256_mul_ps(v1_vec, _mm256_loadu_ps(&v2_start[j + 5*AVX_STRIDE])));
            }
            for (; j <= v2len - AVX_STRIDE; j += AVX_STRIDE) {
                __m256 b = _mm256_loadu_ps(&v2_start[j]);
                _mm256_storeu_ps(&vout_start[row_offset + j], _mm256_mul_ps(v1_vec, b));
            }
            for (; j < v2len; j++) {
                vout_start[row_offset + j] = v1_start[i] * v2_start[j];
            }
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_vector_elementwise_mult -- AVX flat.
 * =========================================================================== */
int vector_vector_elementwise_mult(array_ptr v1, array_ptr v2, array_ptr v_out) {
    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutlen = get_array_length(v_out);

    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_array_start(v_out);

    if (v1len == v2len && v2len == voutlen) {
        int len = v1len;
        int i;
        for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
            __m256 a = _mm256_loadu_ps(&v1_start[i]);
            __m256 b = _mm256_loadu_ps(&v2_start[i]);
            _mm256_storeu_ps(&vout_start[i], _mm256_mul_ps(a, b));
        }
        for (; i < len; i++) {
            vout_start[i] = v1_start[i] * v2_start[i];
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_scalar_mult -- AVX broadcast scalar. Bias parameter update scaling.
 * =========================================================================== */
int vector_scalar_mult(array_ptr v1, data_t scalar, array_ptr v_out) {
    int len     = get_array_length(v1);
    int out_len = get_array_length(v_out);

    data_t* restrict v1_start    = get_array_start(v1);
    data_t* restrict v_out_start = get_array_start(v_out);

    if (len == out_len) {
        __m256 sv = _mm256_set1_ps(scalar);
        int i;
        for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
            __m256 in = _mm256_loadu_ps(&v1_start[i]);
            _mm256_storeu_ps(&v_out_start[i], _mm256_mul_ps(sv, in));
        }
        for (; i < len; i++) {
            v_out_start[i] = scalar * v1_start[i];
        }
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_copy -- AVX flat.
 * =========================================================================== */
int vector_copy(array_ptr source, array_ptr dest) {
    int v1len = get_array_length(source);
    int v2len = get_array_length(dest);

    data_t* restrict v1_start = get_array_start(source);
    data_t* restrict v2_start = get_array_start(dest);

    if (v1len == v2len) {
        int len = v1len;
        int i;
        for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
            _mm256_storeu_ps(&v2_start[i], _mm256_loadu_ps(&v1_start[i]));
        }
        for (; i < len; i++) v2_start[i] = v1_start[i];
        return 1;
    }
    return 0;
}


/* ===========================================================================
 * vector_max -- scalar with hoisted state. Reduction with lane tracking
 * is awkward in AVX; only called TEST_SIZE=10000 times total, left scalar.
 * =========================================================================== */
output_max vector_max(array_ptr v) {
    int len = get_array_length(v);
    data_t* restrict v1_start = get_array_start(v);

    data_t temp_max = v1_start[0];
    int    index    = 0;

    for (int i = 1; i < len; i++) {
        if (v1_start[i] > temp_max) {
            temp_max = v1_start[i];
            index    = i;
        }
    }
    output_max result;
    result.value = temp_max;
    result.index = index;
    return result;
}
