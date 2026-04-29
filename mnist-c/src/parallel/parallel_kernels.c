#include <math.h>
#include <immintrin.h>           /* AVX, AVX2, FMA intrinsics */
#include "parallel_kernels.h"

#define AVX_STRIDE   8           /* 8 floats per 256-bit register */

/* ---------------------------------------------------------------------------
 * Sigmoid uses the "fast sigmoid" / Elliott approximation:
 *     sigmoid(x)  ~ 0.5 + 0.5 * x / (1 + |x|)
 * Derivative of this same function (consistent with the activation):
 *     sigmoid'(x) ~ 0.5 / (1 + |x|)^2
 * Both forms are cheap to vectorize -- no expf() required.
 * --------------------------------------------------------------------------- */

static inline data_t fast_sig(data_t x) {
    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
}

static inline data_t fast_sig_prime(data_t x) {
    data_t denom = 1.0f + fabsf(x);
    return 0.5f / (denom * denom);
}


/* ===========================================================================
 * mat_mat_mult:  C = A x B
 *   A: [m x k], B: [k x n], C: [m x n]
 *
 * Vectorization plan:
 *   - Outer i loop -- one row of C at a time
 *   - Vectorize j-tiles of width 8 (AVX register width)
 *   - 6x unrolled inner r loop with 6 INDEPENDENT accumulator chains
 *     (each c0..c5 holds partial sum for the same 8-wide j-tile from a
 *      different r residue -- summed at the end). This breaks the FMA
 *      latency-throughput dependency chain and lets the CPU keep ~6 FMAs
 *      in flight simultaneously.
 *   - FMA replaces separate mul+add: vfmadd231ps in one instruction.
 *   - Scalar cleanup for j (when n is not a multiple of 8) and for r
 *     (when k is not a multiple of 6).
 *
 * For this network with batched samples, m and n are both modest so the
 * critical dimension is k. H0 layer has k=784, hot path benefits most.
 * =========================================================================== */
void mat_mat_mult(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int m = get_matrix_rows(A);
    long int k = get_matrix_cols(A);
    long int n = get_matrix_cols(B);

    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    for (int i = 0; i < m; i++) {
        long int ai_off = (long int)i * k;
        long int ci_off = (long int)i * n;
        int j;

        /* AVX path: process 8 output columns per j-tile */
        for (j = 0; j <= n - AVX_STRIDE; j += AVX_STRIDE) {

            /* Six independent accumulator chains */
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            __m256 c4 = _mm256_setzero_ps();
            __m256 c5 = _mm256_setzero_ps();

            int r;
            /* 6x unrolled: each iteration consumes 6 r values, one per chain */
            for (r = 0; r <= k - 6; r += 6) {
                __m256 a0 = _mm256_set1_ps(a[ai_off + r    ]);
                __m256 a1 = _mm256_set1_ps(a[ai_off + r + 1]);
                __m256 a2 = _mm256_set1_ps(a[ai_off + r + 2]);
                __m256 a3 = _mm256_set1_ps(a[ai_off + r + 3]);
                __m256 a4 = _mm256_set1_ps(a[ai_off + r + 4]);
                __m256 a5 = _mm256_set1_ps(a[ai_off + r + 5]);

                __m256 b0 = _mm256_loadu_ps(&b[(r    )*n + j]);
                __m256 b1 = _mm256_loadu_ps(&b[(r + 1)*n + j]);
                __m256 b2 = _mm256_loadu_ps(&b[(r + 2)*n + j]);
                __m256 b3 = _mm256_loadu_ps(&b[(r + 3)*n + j]);
                __m256 b4 = _mm256_loadu_ps(&b[(r + 4)*n + j]);
                __m256 b5 = _mm256_loadu_ps(&b[(r + 5)*n + j]);

                c0 = _mm256_fmadd_ps(a0, b0, c0);
                c1 = _mm256_fmadd_ps(a1, b1, c1);
                c2 = _mm256_fmadd_ps(a2, b2, c2);
                c3 = _mm256_fmadd_ps(a3, b3, c3);
                c4 = _mm256_fmadd_ps(a4, b4, c4);
                c5 = _mm256_fmadd_ps(a5, b5, c5);
            }

            /* Pairwise reduction (depth 3 vs linear depth 5) */
            c0 = _mm256_add_ps(c0, c1);
            c2 = _mm256_add_ps(c2, c3);
            c4 = _mm256_add_ps(c4, c5);
            c0 = _mm256_add_ps(c0, c2);
            c0 = _mm256_add_ps(c0, c4);

            /* r cleanup: 0-5 remaining */
            for (; r < k; r++) {
                __m256 ar = _mm256_set1_ps(a[ai_off + r]);
                __m256 br = _mm256_loadu_ps(&b[r*n + j]);
                c0 = _mm256_fmadd_ps(ar, br, c0);
            }

            _mm256_storeu_ps(&c[ci_off + j], c0);
        }

        /* j cleanup: handle the trailing 0-7 columns */
        for (; j < n; j++) {
            data_t sum = 0.0f;
            for (int r = 0; r < k; r++) {
                sum += a[ai_off + r] * b[r*n + j];
            }
            c[ci_off + j] = sum;
        }
    }
}


/* ===========================================================================
 * mat_mat_mult_transB:  C = A x B^T
 *   A: [m x k], B: [n x k] (B^T: [k x n]), C: [m x n]
 *   C[i,j] = sum_r A[i,r] * B[j,r]
 *
 * This is the IDEAL pattern: both A and B are accessed with stride-1 in r.
 * Vectorize the contraction r-loop directly with 6 independent accumulator
 * chains, identical to allTogether's matrix_vector_mult but with an extra
 * outer j loop. Used for weight gradients dW = delta x A_prev^T.
 * =========================================================================== */
void mat_mat_mult_transB(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int m = get_matrix_rows(A);
    long int k = get_matrix_cols(A);
    long int n = get_matrix_rows(B);

    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    for (int i = 0; i < m; i++) {
        long int ai_off = (long int)i * k;
        long int ci_off = (long int)i * n;

        for (int j = 0; j < n; j++) {
            long int bj_off = (long int)j * k;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();

            int r;
            /* 6x unrolled AVX loop: 48 elements per iteration */
            for (r = 0; r <= k - 6*AVX_STRIDE; r += 6*AVX_STRIDE) {
                __m256 a0 = _mm256_loadu_ps(&a[ai_off + r              ]);
                __m256 a1 = _mm256_loadu_ps(&a[ai_off + r +   AVX_STRIDE]);
                __m256 a2 = _mm256_loadu_ps(&a[ai_off + r + 2*AVX_STRIDE]);
                __m256 a3 = _mm256_loadu_ps(&a[ai_off + r + 3*AVX_STRIDE]);
                __m256 a4 = _mm256_loadu_ps(&a[ai_off + r + 4*AVX_STRIDE]);
                __m256 a5 = _mm256_loadu_ps(&a[ai_off + r + 5*AVX_STRIDE]);

                __m256 b0 = _mm256_loadu_ps(&b[bj_off + r              ]);
                __m256 b1 = _mm256_loadu_ps(&b[bj_off + r +   AVX_STRIDE]);
                __m256 b2 = _mm256_loadu_ps(&b[bj_off + r + 2*AVX_STRIDE]);
                __m256 b3 = _mm256_loadu_ps(&b[bj_off + r + 3*AVX_STRIDE]);
                __m256 b4 = _mm256_loadu_ps(&b[bj_off + r + 4*AVX_STRIDE]);
                __m256 b5 = _mm256_loadu_ps(&b[bj_off + r + 5*AVX_STRIDE]);

                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                acc3 = _mm256_fmadd_ps(a3, b3, acc3);
                acc4 = _mm256_fmadd_ps(a4, b4, acc4);
                acc5 = _mm256_fmadd_ps(a5, b5, acc5);
            }

            /* Pairwise reduction */
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc0 = _mm256_add_ps(acc0, acc4);

            /* Single-stride AVX cleanup for the 8-47 trailing range */
            for (; r <= k - AVX_STRIDE; r += AVX_STRIDE) {
                __m256 a0 = _mm256_loadu_ps(&a[ai_off + r]);
                __m256 b0 = _mm256_loadu_ps(&b[bj_off + r]);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            }

            /* Register-based horizontal reduction: ymm -> scalar (no memory) */
            __m128 lo  = _mm256_castps256_ps128(acc0);
            __m128 hi  = _mm256_extractf128_ps(acc0, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            data_t s = _mm_cvtss_f32(sum);

            /* Scalar cleanup for the last 0-7 r values */
            for (; r < k; r++) {
                s += a[ai_off + r] * b[bj_off + r];
            }

            c[ci_off + j] = s;
        }
    }
}


/* ===========================================================================
 * mat_mat_mult_transA:  C = A^T x B
 *   A: [k x m], B: [k x n], C: [m x n]
 *   C[i,j] = sum_r A[r,i] * B[r,j]
 *
 * A is accessed column-wise (stride m in r) -- not vectorizable in r.
 * Vectorize the j dimension instead: for each (r, i), broadcast a[r,i],
 * load 8 elements of b[r,j..j+7], FMA into c[i,j..j+7].
 * Loop order: r outer (so we can amortize a[r,i] broadcast), j inner (AVX).
 * =========================================================================== */
void mat_mat_mult_transA(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int k = get_matrix_rows(A);
    long int m = get_matrix_cols(A);
    long int n = get_matrix_cols(B);

    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    /* Zero C */
    long int total = m * n;
    int idx;
    __m256 zero_v = _mm256_setzero_ps();
    for (idx = 0; idx <= total - AVX_STRIDE; idx += AVX_STRIDE) {
        _mm256_storeu_ps(&c[idx], zero_v);
    }
    for (; idx < total; idx++) c[idx] = 0.0f;

    /* Accumulate */
    for (int r = 0; r < k; r++) {
        long int ar_off = (long int)r * m;
        long int br_off = (long int)r * n;
        for (int i = 0; i < m; i++) {
            __m256 a_bcast = _mm256_set1_ps(a[ar_off + i]);
            long int ci_off = (long int)i * n;
            int j;
            for (j = 0; j <= n - AVX_STRIDE; j += AVX_STRIDE) {
                __m256 b_vec = _mm256_loadu_ps(&b[br_off + j]);
                __m256 c_vec = _mm256_loadu_ps(&c[ci_off + j]);
                c_vec = _mm256_fmadd_ps(a_bcast, b_vec, c_vec);
                _mm256_storeu_ps(&c[ci_off + j], c_vec);
            }
            for (; j < n; j++) {
                c[ci_off + j] += a[ar_off + i] * b[br_off + j];
            }
        }
    }
}


/* ===========================================================================
 * mat_add_bias:  Z[i,j] += b[i] for all j  (broadcast bias across columns)
 *   Vectorize j with AVX, broadcast scalar b[i] into all 8 lanes.
 * =========================================================================== */
void mat_add_bias(matrix_ptr Z, array_ptr b) {
    long int rows = get_matrix_rows(Z);
    long int cols = get_matrix_cols(Z);
    data_t* restrict z  = get_matrix_start(Z);
    data_t* restrict bv = get_array_start(b);

    for (int i = 0; i < rows; i++) {
        long int row_off = (long int)i * cols;
        __m256 b_vec = _mm256_set1_ps(bv[i]);
        int j;
        for (j = 0; j <= cols - AVX_STRIDE; j += AVX_STRIDE) {
            __m256 z_vec = _mm256_loadu_ps(&z[row_off + j]);
            _mm256_storeu_ps(&z[row_off + j], _mm256_add_ps(z_vec, b_vec));
        }
        for (; j < cols; j++) {
            z[row_off + j] += bv[i];
        }
    }
}


/* ===========================================================================
 * mat_sigmoid_inplace:  fast sigmoid 0.5 + 0.5*x/(1+|x|)
 *   Avoids expf(); uses andnot for absolute value.
 * =========================================================================== */
void mat_sigmoid_inplace(matrix_ptr A) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t* restrict a = get_matrix_start(A);

    __m256 half      = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);   /* 0x80000000 -- the sign bit only */

    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
        __m256 x       = _mm256_loadu_ps(&a[i]);
        __m256 abs_x   = _mm256_andnot_ps(sign_mask, x);   /* |x| */
        __m256 denom   = _mm256_add_ps(one, abs_x);        /* 1 + |x| */
        __m256 ratio   = _mm256_div_ps(x, denom);          /* x / (1+|x|) */
        __m256 scaled  = _mm256_mul_ps(half, ratio);       /* 0.5 * ... */
        __m256 result  = _mm256_add_ps(half, scaled);      /* 0.5 + ... */
        _mm256_storeu_ps(&a[i], result);
    }
    for (; i < total; i++) {
        a[i] = fast_sig(a[i]);
    }
}


/* ===========================================================================
 * mat_sigmoid_prime:  derivative of fast sigmoid
 *     sigma'(x) = 0.5 / (1 + |x|)^2
 *   Consistent with mat_sigmoid_inplace above.
 * =========================================================================== */
void mat_sigmoid_prime(matrix_ptr Z, matrix_ptr Out) {
    long int total = get_matrix_rows(Z) * get_matrix_cols(Z);
    data_t* restrict z   = get_matrix_start(Z);
    data_t* restrict out = get_matrix_start(Out);

    __m256 half      = _mm256_set1_ps(0.5f);
    __m256 one       = _mm256_set1_ps(1.0f);
    __m256 sign_mask = _mm256_set1_ps(-0.0f);

    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
        __m256 x       = _mm256_loadu_ps(&z[i]);
        __m256 abs_x   = _mm256_andnot_ps(sign_mask, x);
        __m256 denom   = _mm256_add_ps(one, abs_x);
        __m256 denom2  = _mm256_mul_ps(denom, denom);
        __m256 result  = _mm256_div_ps(half, denom2);
        _mm256_storeu_ps(&out[i], result);
    }
    for (; i < total; i++) {
        out[i] = fast_sig_prime(z[i]);
    }
}


/* ===========================================================================
 * mat_hadamard:  C[i,j] = A[i,j] * B[i,j]
 *   Element-wise multiply over flat 1D layout, simple AVX (no FMA pattern).
 * =========================================================================== */
void mat_hadamard(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(av, bv));
    }
    for (; i < total; i++) c[i] = a[i] * b[i];
}


/* ===========================================================================
 * mat_sub:  C[i,j] = A[i,j] - B[i,j]
 * =========================================================================== */
void mat_sub(matrix_ptr A, matrix_ptr B, matrix_ptr C) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);
    data_t* restrict c = get_matrix_start(C);

    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_sub_ps(av, bv));
    }
    for (; i < total; i++) c[i] = a[i] - b[i];
}


/* ===========================================================================
 * mat_row_sum:  v[i] = sum_j A[i,j]
 *   For each row, vectorize the column reduction with 6 independent
 *   accumulator chains and a register-based horizontal reduction.
 * =========================================================================== */
void mat_row_sum(matrix_ptr A, array_ptr v) {
    long int rows = get_matrix_rows(A);
    long int cols = get_matrix_cols(A);
    data_t* restrict a  = get_matrix_start(A);
    data_t* restrict vd = get_array_start(v);

    for (int i = 0; i < rows; i++) {
        long int row_off = (long int)i * cols;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();

        int j;
        for (j = 0; j <= cols - 6*AVX_STRIDE; j += 6*AVX_STRIDE) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&a[row_off + j              ]));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(&a[row_off + j +   AVX_STRIDE]));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(&a[row_off + j + 2*AVX_STRIDE]));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(&a[row_off + j + 3*AVX_STRIDE]));
            acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps(&a[row_off + j + 4*AVX_STRIDE]));
            acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps(&a[row_off + j + 5*AVX_STRIDE]));
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc0 = _mm256_add_ps(acc0, acc2);
        acc0 = _mm256_add_ps(acc0, acc4);

        for (; j <= cols - AVX_STRIDE; j += AVX_STRIDE) {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(&a[row_off + j]));
        }

        __m128 lo  = _mm256_castps256_ps128(acc0);
        __m128 hi  = _mm256_extractf128_ps(acc0, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        data_t s = _mm_cvtss_f32(sum);

        for (; j < cols; j++) s += a[row_off + j];

        vd[i] = s;
    }
}


/* ===========================================================================
 * mat_copy:  flat 1D AVX load/store
 * =========================================================================== */
void mat_copy(matrix_ptr A, matrix_ptr B) {
    long int total = get_matrix_rows(A) * get_matrix_cols(A);
    data_t* restrict a = get_matrix_start(A);
    data_t* restrict b = get_matrix_start(B);

    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
        _mm256_storeu_ps(&b[i], _mm256_loadu_ps(&a[i]));
    }
    for (; i < total; i++) b[i] = a[i];
}
