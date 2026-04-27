#include "kernels.h"
#include "array_matrix_funcs.h"
#include <immintrin.h>          /* AVX, AVX2, FMA, SSE3 intrinsics      */

#define AVX_STRIDE   8          /* floats per 256-bit register           */
#define AVX_STRIDE_6 48         /* 6 * AVX_STRIDE: floats per unrolled iter */

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

            /* Six independent accumulator chains.
             * acc0: positions  0, 48,  96, ...
             * acc1: positions  8, 56, 104, ...
             * acc2: positions 16, 64, 112, ...
             * acc3: positions 24, 72, 120, ...
             * acc4: positions 32, 80, 128, ...
             * acc5: positions 40, 88, 136, ...                          */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();

            int j;

            /* --- 6x Unrolled AVX+FMA loop: 48 elements per iteration ---
             * CHANGE 1: _mm256_fmadd_ps(w, a, acc) replaces the separate
             * _mm256_mul_ps + _mm256_add_ps pair in every chain.
             * This halves the instruction count for the arithmetic:
             *   Before: 2 instructions (vmulps + vaddps) per chain
             *   After:  1 instruction  (vfmadd231ps)     per chain
             * Same 4-cycle latency as vaddps, same 0.5-cycle throughput,
             * but frees up port bandwidth previously used by vmulps.
             * For cols=784: runs j=0,48,...,720 (16 full iterations).   */
            for (j = 0; j < cols - (AVX_STRIDE_6 - 1); j += AVX_STRIDE_6) {

                /* Chain 0: elements j .. j+7 */
                __m256 w0 = _mm256_loadu_ps(&weights[row_offset + j]);
                __m256 a0 = _mm256_loadu_ps(&lastLayerActivations[j]);
                acc0 = _mm256_fmadd_ps(w0, a0, acc0);      /* w0*a0 + acc0 */

                /* Chain 1: elements j+8 .. j+15 (independent of chain 0) */
                __m256 w1 = _mm256_loadu_ps(&weights[row_offset + j + AVX_STRIDE]);
                __m256 a1 = _mm256_loadu_ps(&lastLayerActivations[j + AVX_STRIDE]);
                acc1 = _mm256_fmadd_ps(w1, a1, acc1);      /* w1*a1 + acc1 */

                /* Chain 2: elements j+16 .. j+23 (independent of 0,1)   */
                __m256 w2 = _mm256_loadu_ps(&weights[row_offset + j + 2*AVX_STRIDE]);
                __m256 a2 = _mm256_loadu_ps(&lastLayerActivations[j + 2*AVX_STRIDE]);
                acc2 = _mm256_fmadd_ps(w2, a2, acc2);      /* w2*a2 + acc2 */

                /* Chain 3: elements j+24 .. j+31 (independent of 0,1,2) */
                __m256 w3 = _mm256_loadu_ps(&weights[row_offset + j + 3*AVX_STRIDE]);
                __m256 a3 = _mm256_loadu_ps(&lastLayerActivations[j + 3*AVX_STRIDE]);
                acc3 = _mm256_fmadd_ps(w3, a3, acc3);      /* w3*a3 + acc3 */

                /* Chain 4: elements j+32 .. j+39 (independent of 0-3)   */
                __m256 w4 = _mm256_loadu_ps(&weights[row_offset + j + 4*AVX_STRIDE]);
                __m256 a4 = _mm256_loadu_ps(&lastLayerActivations[j + 4*AVX_STRIDE]);
                acc4 = _mm256_fmadd_ps(w4, a4, acc4);      /* w4*a4 + acc4 */

                /* Chain 5: elements j+40 .. j+47 (independent of 0-4)   */
                __m256 w5 = _mm256_loadu_ps(&weights[row_offset + j + 5*AVX_STRIDE]);
                __m256 a5 = _mm256_loadu_ps(&lastLayerActivations[j + 5*AVX_STRIDE]);
                acc5 = _mm256_fmadd_ps(w5, a5, acc5);      /* w5*a5 + acc5 */
            }

            /* Pairwise merge of 6 accumulators — depth 3 vs linear depth 5.
             * Level 1: three independent adds execute simultaneously     */
            acc0 = _mm256_add_ps(acc0, acc1);   /* independent } level 1 */
            acc2 = _mm256_add_ps(acc2, acc3);   /* independent } level 1 */
            acc4 = _mm256_add_ps(acc4, acc5);   /* independent } level 1 */
            acc0 = _mm256_add_ps(acc0, acc2);   /*               level 2 */
            acc0 = _mm256_add_ps(acc0, acc4);   /*               level 3 */

            /* --- Single-stride AVX+FMA fallthrough: handles 8-47 remainder ---
             * CHANGE 1 applied here too: fmadd replaces mul+add.
             * For cols=784: fires at j=768 and j=776, consuming all 16. */
            for (; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 w = _mm256_loadu_ps(&weights[row_offset + j]);
                __m256 a = _mm256_loadu_ps(&lastLayerActivations[j]);
                acc0 = _mm256_fmadd_ps(w, a, acc0);
            }

            /* --- CHANGE 2: Register-based horizontal reduction ---
             * Replaces: store acc0 to temp[8], sum 8 scalars (touches memory)
             * With:     pure register operations using extractf128 + hadd
             *
             * Step 1: fold the upper 128 bits down onto the lower 128 bits.
             *   castps256_ps128: free operation, no instruction generated —
             *     just reinterprets the lower half of the ymm register as xmm.
             *   extractf128: one instruction (vextractf128) to pull upper half.
             *   _mm_add_ps:  [s0+s4, s1+s5, s2+s6, s3+s7] in one xmm reg.  */
            __m128 lo128  = _mm256_castps256_ps128(acc0);   /* free: ymm->xmm */
            __m128 hi128  = _mm256_extractf128_ps(acc0, 1); /* upper 128 bits */
            __m128 sum128 = _mm_add_ps(lo128, hi128);       /* 4 partial sums */

            /* Step 2: two horizontal adds collapse 4 lanes to 1.
             *   First  hadd: [s0+s4+s1+s5, s2+s6+s3+s7, s0+s4+s1+s5, ...]  */
            sum128 = _mm_hadd_ps(sum128, sum128);
            /*   Second hadd: [total, total, total, total]                    */
            sum128 = _mm_hadd_ps(sum128, sum128);

            /* Step 3: extract scalar from lane 0 — no memory touched        */
            data_t sum = _mm_cvtss_f32(sum128);

            /* --- Scalar cleanup: 0-7 remaining elements ---
             * For cols=784: j=784, never fires.                         */
            for (; j < cols; j++) {
                sum += weights[row_offset + j] * lastLayerActivations[j];
            }

            v_out_loc[i] = sum;
        }
        return 1;
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * kernel_vector_vector_mult — UNCHANGED
 * No multiply-accumulate pattern: FMA does not apply.
 * No lane reduction: hadd does not apply.
 * ----------------------------------------------------------------------- */
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

            for (; j < v2len; j++) {
                vout_start[row_offset + j] = v1_val * v2_start[j];
            }
        }
        return 1;
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * kernel_matrix_matrix_add — UNCHANGED
 * Add-only pattern: FMA requires a multiply, does not apply.
 * No lane reduction: hadd does not apply.
 * ----------------------------------------------------------------------- */
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
                _mm256_storeu_ps(&m_out_start[row_offset + j],
                                 _mm256_add_ps(a0, b0));

                __m256 a1 = _mm256_loadu_ps(&m1_start[row_offset + j + AVX_STRIDE]);
                __m256 b1 = _mm256_loadu_ps(&m2_start[row_offset + j + AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + AVX_STRIDE],
                                 _mm256_add_ps(a1, b1));

                __m256 a2 = _mm256_loadu_ps(&m1_start[row_offset + j + 2*AVX_STRIDE]);
                __m256 b2 = _mm256_loadu_ps(&m2_start[row_offset + j + 2*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 2*AVX_STRIDE],
                                 _mm256_add_ps(a2, b2));

                __m256 a3 = _mm256_loadu_ps(&m1_start[row_offset + j + 3*AVX_STRIDE]);
                __m256 b3 = _mm256_loadu_ps(&m2_start[row_offset + j + 3*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 3*AVX_STRIDE],
                                 _mm256_add_ps(a3, b3));

                __m256 a4 = _mm256_loadu_ps(&m1_start[row_offset + j + 4*AVX_STRIDE]);
                __m256 b4 = _mm256_loadu_ps(&m2_start[row_offset + j + 4*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 4*AVX_STRIDE],
                                 _mm256_add_ps(a4, b4));

                __m256 a5 = _mm256_loadu_ps(&m1_start[row_offset + j + 5*AVX_STRIDE]);
                __m256 b5 = _mm256_loadu_ps(&m2_start[row_offset + j + 5*AVX_STRIDE]);
                _mm256_storeu_ps(&m_out_start[row_offset + j + 5*AVX_STRIDE],
                                 _mm256_add_ps(a5, b5));
            }

            for (; j < cols1 - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 a = _mm256_loadu_ps(&m1_start[row_offset + j]);
                __m256 b = _mm256_loadu_ps(&m2_start[row_offset + j]);
                _mm256_storeu_ps(&m_out_start[row_offset + j],
                                 _mm256_add_ps(a, b));
            }

            for (; j < cols1; j++) {
                m_out_start[row_offset + j] =
                    m1_start[row_offset + j] + m2_start[row_offset + j];
            }
        }
        return 1;
    }

    return 0;
}