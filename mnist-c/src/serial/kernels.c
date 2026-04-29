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


int kernel_vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out) {

    int v1len   = get_array_length(v1);
    int v2len   = get_array_length(v2);
    int voutlen = get_array_length(v_out);

    /* restrict eliminates aliasing between all three pointers            */
    data_t* restrict v1_start   = get_array_start(v1);
    data_t* restrict v2_start   = get_array_start(v2);
    data_t* restrict vout_start = get_array_start(v_out);

    if (v1len == v2len && v2len == voutlen) {

        /* Single length variable — unambiguous loop bound after
         * the equality check confirms all three lengths are equal        */
        int len = v1len;

        int i;

        /* --- AVX loop: process 8 elements per iteration ---
         * This is the simplest AVX pattern possible:
         *   - No reduction:  results written directly to vout (unlike
         *                    matrix_vector_mult which collapses to scalar)
         *   - No broadcast:  all three arrays vary each iteration (unlike
         *                    vector_vector_mult which broadcasts v1[i])
         *   - Flat 1D:       no row_offset calculation needed (unlike
         *                    matrix_matrix_add which has 2D indexing)
         * Bound (len - 7) ensures indices i through i+7 are always valid.
         * For len=784: covers all 784 elements, cleanup never fires.
         * For len=16:  covers all 16 elements,  cleanup never fires.     */
        for (i = 0; i < len - (AVX_STRIDE - 1); i += AVX_STRIDE) {

            /* Load 8 elements from v1: v1[i] .. v1[i+7]                 */
            __m256 a = _mm256_loadu_ps(&v1_start[i]);

            /* Load 8 elements from v2: v2[i] .. v2[i+7]                 */
            __m256 b = _mm256_loadu_ps(&v2_start[i]);

            /* Element-wise add: lane k = v1[i+k] + v2[i+k]              */
            __m256 c = _mm256_add_ps(a, b);

            /* Store 8 results directly to vout: no reduction needed      */
            _mm256_storeu_ps(&vout_start[i], c);
        }

        /* --- Scalar cleanup: handles remaining 0-7 elements ---
         * For len=784: i=784, never fires.
         * For len=100: i=96,  fires 4 times (i=96,97,98,99).
         * For len=16:  i=16,  never fires.
         * For len=10:  i=8,   fires 2 times (i=8,9).                    */
        for (; i < len; i++) {
            vout_start[i] = v1_start[i] + v2_start[i];
        }

        return 1;
    }

    return 0;
}


int kernel_matrix_transpose(matrix_ptr m, matrix_ptr m_out) {

    long int rows     = get_matrix_rows(m);
    long int cols     = get_matrix_cols(m);
    long int out_rows = get_matrix_rows(m_out);
    long int out_cols = get_matrix_cols(m_out);

    /* CHANGE 1: restrict on both pointers — removes aliasing between
     * input and output, allowing the compiler to keep loaded values
     * in registers rather than reloading from memory each iteration.   */
    data_t* restrict m_start     = get_matrix_start(m);
    data_t* restrict m_out_start = get_matrix_start(m_out);

    if (rows == out_cols && cols == out_rows) {

        /* CHANGE 2: precompute stride multiples once outside both loops.
         * In the inner loop, moving one step in j moves out_cols positions
         * forward in the output. For the unrolled offsets (j+1, j+2 ... j+5),
         * we need 1×, 2×, 3×, 4×, 5× out_cols as offsets from the base
         * output index. Computing these here saves 5 multiplications per
         * outer iteration (100 outer iterations × 5 = 500 multiplications
         * saved for the 100×784 matrix).                                */
        long int s1 = out_cols;
        long int s2 = 2L * out_cols;
        long int s3 = 3L * out_cols;
        long int s4 = 4L * out_cols;
        long int s5 = 5L * out_cols;
        long int s6 = 6L * out_cols;   /* stride for advancing out_idx  */

        int i, j;

        for (i = 0; i < rows; i++) {

            /* CHANGE 3: hoist input row offset — same as matrix_vector_mult.
             * Avoids recomputing i*cols on every inner iteration.         */
            long int row_offset = (long int)i * cols;

            /* CHANGE 4: incremental output index replaces j*out_cols + i.
             * Original: m_out_start[j*out_cols + i] — 1 multiply per step.
             * Optimized: out_idx starts at column i of the output (row 0),
             * and advances by out_cols each j step — 1 add per step.
             * Invariant: out_idx == i + j*out_cols at all times.          */
            long int out_idx = (long int)i;

            /* CHANGE 5: 6x unrolled inner loop.
             * Each iteration reads 6 consecutive elements from input row i
             * and writes them to 6 consecutive rows of output column i.
             * The precomputed s1..s5 offsets index those 6 output rows
             * from the base out_idx without any multiplication.
             * out_idx advances by s6 = 6*out_cols at the end of each
             * iteration to stay aligned with j.
             * Bound (cols-5) ensures j+5 is always a valid index.
             * For cols=784: runs j=0,6,...,774 (130 iters, covers 0-779),
             * leaving 4 elements (j=780..783) for the cleanup loop.      */
            for (j = 0; j < cols - 5; j += 6, out_idx += s6) {
                m_out_start[out_idx]      = m_start[row_offset + j];
                m_out_start[out_idx + s1] = m_start[row_offset + j + 1];
                m_out_start[out_idx + s2] = m_start[row_offset + j + 2];
                m_out_start[out_idx + s3] = m_start[row_offset + j + 3];
                m_out_start[out_idx + s4] = m_start[row_offset + j + 4];
                m_out_start[out_idx + s5] = m_start[row_offset + j + 5];
            }

            /* Scalar cleanup: handles remaining 0-5 elements.
             * out_idx is correctly positioned at i + j*out_cols here
             * since it was incremented by s6 in sync with j.
             * For cols=784: j=780, fires 4 times (j=780,781,782,783).
             * For cols=100: j= 96, fires 4 times (j=96,97,98,99).       */
            for (; j < cols; j++, out_idx += out_cols) {
                m_out_start[out_idx] = m_start[row_offset + j];
            }
        }
        return 1;
    }

    return 0;
}

int kernel_matrix_scalar_mult(matrix_ptr m, data_t scalar, matrix_ptr m_out) {
    long int rows = get_matrix_rows(m);
    long int cols = get_matrix_cols(m);
    long int out_rows = get_matrix_rows(m_out);
    long int out_cols = get_matrix_cols(m_out);

    data_t* m_start = get_matrix_start(m);
    data_t* m_out_start = get_matrix_start(m_out);

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    int i, j;

    if(rows == out_rows && cols == out_cols){
        for(i = 0; i < rows; i++) {
            int row_offset = i*cols;

            for (j = 0; j < cols - (AVX_STRIDE - 1); j += AVX_STRIDE) {
                __m256 in = _mm256_loadu_ps(&m_start[row_offset + j]);

                __m256 out = _mm256_mul_ps(scalar_vec, in);

                _mm256_storeu_ps(&m_out_start[row_offset + j], out);
            }

            for (; j < cols; j++) {
                m_out_start[row_offset + j] = scalar * m_start[row_offset + j];
            }
        }
        return 1;
    }
    return 0;   
}