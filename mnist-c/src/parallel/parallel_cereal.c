#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../params.h"
#include "../serial/array_matrix_funcs.h"
#include "../serial/array_matrix_math.h"
#include "parallel_kernels.h"
#include "parallel_cereal.h"

#define I_SIZE   IMAGE_SIZE   // 784
#define H0_SIZE  100
#define H1_SIZE  16
#define L_SIZE   10
#define BATCH_SIZE  100
#define LEARN_RATE  3
#define NUM_EPOCHS  10

// ─── Shared network parameters (read-only inside the parallel region) ────────
// Written only during parameter update, which runs serially after each batch.

static matrix_ptr H0_W, H1_W, L_W;
static array_ptr  H0_B, H1_B, L_B;

// ─── Per-thread buffers (allocated as arrays of length num_threads) ───────────
//
// Layout convention: all sub-batch matrices are [layer_size x S] where S is
// the sub-batch size. Columns correspond to samples, rows to neurons/features.
// This lets the forward GEMM be:  Z = W x X  ([neurons x S] = [neurons x features] x [features x S])
// and the backward weight GEMM:  dW = delta x X^T  ([neurons_out x neurons_in] = ... )

static matrix_ptr *IN_t;          // [I_SIZE  x S]  -- sub-batch inputs

static matrix_ptr *H0_Z_t;        // [H0_SIZE x S]  -- pre-activation (saved for sigmoid' in backprop)
static matrix_ptr *H0_A_t;        // [H0_SIZE x S]  -- activations = sigmoid(H0_Z)
static matrix_ptr *H1_Z_t;        // [H1_SIZE x S]
static matrix_ptr *H1_A_t;        // [H1_SIZE x S]
static matrix_ptr *OUT_Z_t;       // [L_SIZE  x S]
static matrix_ptr *OUT_A_t;       // [L_SIZE  x S]

static matrix_ptr *Y_t;           // [L_SIZE  x S]  -- one-hot label matrix for sub-batch

// Backprop scratch: deltas and error propagation buffers
static matrix_ptr *delta_L_t;     // [L_SIZE  x S]
static matrix_ptr *delta_H1_t;    // [H1_SIZE x S]
static matrix_ptr *delta_H0_t;    // [H0_SIZE x S]
static matrix_ptr *err_H1_t;      // [H1_SIZE x S]  -- W_L^T x delta_L, also temp for sigmoid'(OUT_Z)
static matrix_ptr *err_H0_t;      // [H0_SIZE x S]  -- W_H1^T x delta_H1

// Per-thread gradient outputs (summed over the sub-batch via GEMM, then reduced across threads)
static matrix_ptr *H0_W_g_t;      // [H0_SIZE x I_SIZE]
static matrix_ptr *H1_W_g_t;      // [H1_SIZE x H0_SIZE]
static matrix_ptr *L_W_g_t;       // [L_SIZE  x H1_SIZE]
static array_ptr  *H0_B_g_t;      // [H0_SIZE]
static array_ptr  *H1_B_g_t;      // [H1_SIZE]
static array_ptr  *L_B_g_t;       // [L_SIZE]

static int sub_batch_size;        // BATCH_SIZE / num_threads


// ─── Initialization ───────────────────────────────────────────────────────────

static void init_parallel_MNIST(int num_threads) {
    sub_batch_size = BATCH_SIZE / num_threads;
    int S = sub_batch_size;

    // Shared weights -- Xavier uniform: range +/- sqrt(1 / fan_in)
    H0_W = new_matrix(H0_SIZE, I_SIZE);
    H0_B = new_array(H0_SIZE);
    init_matrix_rand(H0_W, -sqrt(1.0 / I_SIZE), sqrt(1.0 / I_SIZE));

    H1_W = new_matrix(H1_SIZE, H0_SIZE);
    H1_B = new_array(H1_SIZE);
    init_matrix_rand(H1_W, -sqrt(1.0 / H0_SIZE), sqrt(1.0 / H0_SIZE));

    L_W = new_matrix(L_SIZE, H1_SIZE);
    L_B = new_array(L_SIZE);
    init_matrix_rand(L_W, -sqrt(1.0 / H1_SIZE), sqrt(1.0 / H1_SIZE));

    // Allocate per-thread pointer arrays
    IN_t      = malloc(num_threads * sizeof(matrix_ptr));
    H0_Z_t    = malloc(num_threads * sizeof(matrix_ptr));
    H0_A_t    = malloc(num_threads * sizeof(matrix_ptr));
    H1_Z_t    = malloc(num_threads * sizeof(matrix_ptr));
    H1_A_t    = malloc(num_threads * sizeof(matrix_ptr));
    OUT_Z_t   = malloc(num_threads * sizeof(matrix_ptr));
    OUT_A_t   = malloc(num_threads * sizeof(matrix_ptr));
    Y_t       = malloc(num_threads * sizeof(matrix_ptr));
    delta_L_t  = malloc(num_threads * sizeof(matrix_ptr));
    delta_H1_t = malloc(num_threads * sizeof(matrix_ptr));
    delta_H0_t = malloc(num_threads * sizeof(matrix_ptr));
    err_H1_t   = malloc(num_threads * sizeof(matrix_ptr));
    err_H0_t   = malloc(num_threads * sizeof(matrix_ptr));
    H0_W_g_t  = malloc(num_threads * sizeof(matrix_ptr));
    H1_W_g_t  = malloc(num_threads * sizeof(matrix_ptr));
    L_W_g_t   = malloc(num_threads * sizeof(matrix_ptr));
    H0_B_g_t  = malloc(num_threads * sizeof(array_ptr));
    H1_B_g_t  = malloc(num_threads * sizeof(array_ptr));
    L_B_g_t   = malloc(num_threads * sizeof(array_ptr));

    // Allocate per-thread matrices
    for (int t = 0; t < num_threads; t++) {
        IN_t[t]      = new_matrix(I_SIZE,  S);
        H0_Z_t[t]    = new_matrix(H0_SIZE, S);
        H0_A_t[t]    = new_matrix(H0_SIZE, S);
        H1_Z_t[t]    = new_matrix(H1_SIZE, S);
        H1_A_t[t]    = new_matrix(H1_SIZE, S);
        OUT_Z_t[t]   = new_matrix(L_SIZE,  S);
        OUT_A_t[t]   = new_matrix(L_SIZE,  S);
        Y_t[t]       = new_matrix(L_SIZE,  S);
        delta_L_t[t]  = new_matrix(L_SIZE,  S);
        delta_H1_t[t] = new_matrix(H1_SIZE, S);
        delta_H0_t[t] = new_matrix(H0_SIZE, S);
        err_H1_t[t]   = new_matrix(H1_SIZE, S);
        err_H0_t[t]   = new_matrix(H0_SIZE, S);
        H0_W_g_t[t]  = new_matrix(H0_SIZE, I_SIZE);
        H1_W_g_t[t]  = new_matrix(H1_SIZE, H0_SIZE);
        L_W_g_t[t]   = new_matrix(L_SIZE,  H1_SIZE);
        H0_B_g_t[t]  = new_array(H0_SIZE);
        H1_B_g_t[t]  = new_array(H1_SIZE);
        L_B_g_t[t]   = new_array(L_SIZE);
    }
}


// ─── Sub-batch data loading ───────────────────────────────────────────────────
//
// Copies S images into columns of IN_t[tid] and builds the one-hot label
// matrix Y_t[tid].  The column layout means IN_t[feature, sample] is stored
// at IN_t->data[feature * S + sample].

static void load_sub_batch(dataset_ptr data, int tid, int *indices, int batch_start) {
    int S = sub_batch_size;
    data_t *in_data = get_matrix_start(IN_t[tid]);
    data_t *y_data  = get_matrix_start(Y_t[tid]);
    int sub_start   = batch_start + tid * S;

    // Zero Y -- labels differ every batch
    zero_matrix(Y_t[tid]);

    for (int s = 0; s < S; s++) {
        int img_idx = indices[sub_start + s];
        int label   = data->nums[img_idx];

        // Copy image pixels into column s of IN_t
        for (int f = 0; f < I_SIZE; f++) {
            in_data[f * S + s] = data->image_arr[img_idx * I_SIZE + f];
        }

        // Set one-hot entry for this sample's label
        y_data[label * S + s] = 1.0f;
    }
}


// ─── Batched feedforward ─────────────────────────────────────────────────────
//
// For each layer:
//   Z = W x A_prev          (GEMM: [neurons x S] = [neurons x prev] x [prev x S])
//   Z += B                  (broadcast bias across S sample columns)
//   A = sigmoid(Z)          (element-wise; Z is preserved for sigmoid' in backprop)

static void batched_feedforward(int tid) {

    // Layer 0: IN -> H0
    mat_mat_mult(H0_W, IN_t[tid], H0_Z_t[tid]);      // H0_Z = H0_W x IN   [100xS]
    mat_add_bias(H0_Z_t[tid], H0_B);                  // H0_Z += H0_B
    mat_copy(H0_Z_t[tid], H0_A_t[tid]);               // H0_A = H0_Z (copy before sigmoid)
    mat_sigmoid_inplace(H0_A_t[tid]);                  // H0_A = sigmoid(H0_A)

    // Layer 1: H0 -> H1
    mat_mat_mult(H1_W, H0_A_t[tid], H1_Z_t[tid]);    // H1_Z = H1_W x H0_A  [16xS]
    mat_add_bias(H1_Z_t[tid], H1_B);
    mat_copy(H1_Z_t[tid], H1_A_t[tid]);
    mat_sigmoid_inplace(H1_A_t[tid]);

    // Layer 2: H1 -> OUT
    mat_mat_mult(L_W, H1_A_t[tid], OUT_Z_t[tid]);    // OUT_Z = L_W x H1_A  [10xS]
    mat_add_bias(OUT_Z_t[tid], L_B);
    mat_copy(OUT_Z_t[tid], OUT_A_t[tid]);
    mat_sigmoid_inplace(OUT_A_t[tid]);
}


// ─── Batched backpropagation ──────────────────────────────────────────────────
//
// For each layer (output -> input):
//   delta   = error_from_next ⊙ sigmoid'(Z)          (Hadamard with sigmoid derivative)
//   dW      = delta x A_prev^T                        (GEMM: weight gradient, sum over S samples)
//   dB      = rowsum(delta)                           (bias gradient, sum over S samples)
//   err_prev = W^T x delta                            (error propagated to previous layer)
//
// The three serial hotspots (matrix-vector, outer product, accumulation) are
// replaced by two GEMMs per layer.  Gradient accumulation over S samples is
// implicit in the GEMM inner loop rather than a separate pass.

static void batched_backprop(int tid) {

    // === OUTPUT LAYER ===

    // delta_L = (OUT_A - Y) ⊙ sigmoid'(OUT_Z)
    // Re-use err_H1_t as scratch for sigmoid'(OUT_Z) -- it gets overwritten below anyway.
    mat_sub(OUT_A_t[tid], Y_t[tid], delta_L_t[tid]);         // delta_L = OUT_A - Y
    mat_sigmoid_prime(OUT_Z_t[tid], err_H1_t[tid]);           // err_H1 (temp) = sigmoid'(OUT_Z)
    mat_hadamard(delta_L_t[tid], err_H1_t[tid], delta_L_t[tid]); // delta_L ⊙= sigmoid'

    // dW_L = delta_L x H1_A^T   [L_SIZE x S] x [S x H1_SIZE] -> [L_SIZE x H1_SIZE]
    mat_mat_mult_transB(delta_L_t[tid], H1_A_t[tid], L_W_g_t[tid]);

    // dB_L = rowsum(delta_L)  -> [L_SIZE]
    mat_row_sum(delta_L_t[tid], L_B_g_t[tid]);

    // === HIDDEN LAYER 1 ===

    // err_H1 = L_W^T x delta_L   [H1_SIZE x L_SIZE] x [L_SIZE x S] -> [H1_SIZE x S]
    // (L_W is [L_SIZE x H1_SIZE]; transA gives us L_W^T x delta_L)
    mat_mat_mult_transA(L_W, delta_L_t[tid], err_H1_t[tid]);

    // delta_H1 = err_H1 ⊙ sigmoid'(H1_Z)
    // Compute sigmoid'(H1_Z) directly into delta_H1_t, then hadamard with err_H1_t.
    mat_sigmoid_prime(H1_Z_t[tid], delta_H1_t[tid]);          // delta_H1 = sigmoid'(H1_Z)
    mat_hadamard(err_H1_t[tid], delta_H1_t[tid], delta_H1_t[tid]); // delta_H1 = err ⊙ sigmoid'

    // dW_H1 = delta_H1 x H0_A^T  [H1_SIZE x S] x [S x H0_SIZE] -> [H1_SIZE x H0_SIZE]
    mat_mat_mult_transB(delta_H1_t[tid], H0_A_t[tid], H1_W_g_t[tid]);

    // dB_H1 = rowsum(delta_H1)  -> [H1_SIZE]
    mat_row_sum(delta_H1_t[tid], H1_B_g_t[tid]);

    // === HIDDEN LAYER 0 ===

    // err_H0 = H1_W^T x delta_H1  [H0_SIZE x H1_SIZE] x [H1_SIZE x S] -> [H0_SIZE x S]
    // (H1_W is [H1_SIZE x H0_SIZE]; transA gives H1_W^T x delta_H1)
    mat_mat_mult_transA(H1_W, delta_H1_t[tid], err_H0_t[tid]);

    // delta_H0 = err_H0 ⊙ sigmoid'(H0_Z)
    mat_sigmoid_prime(H0_Z_t[tid], delta_H0_t[tid]);          // delta_H0 = sigmoid'(H0_Z)
    mat_hadamard(err_H0_t[tid], delta_H0_t[tid], delta_H0_t[tid]); // delta_H0 = err ⊙ sigmoid'

    // dW_H0 = delta_H0 x IN^T   [H0_SIZE x S] x [S x I_SIZE] -> [H0_SIZE x I_SIZE]
    // This single GEMM replaces both the outer-product (hotspot 2) and
    // the per-sample accumulation loop (hotspot 3) from the serial version.
    mat_mat_mult_transB(delta_H0_t[tid], IN_t[tid], H0_W_g_t[tid]);

    // dB_H0 = rowsum(delta_H0)  -> [H0_SIZE]
    mat_row_sum(delta_H0_t[tid], H0_B_g_t[tid]);
}


// ─── Training loop ────────────────────────────────────────────────────────────

static void train_parallel_MNIST(dataset_ptr train_data, int num_threads) {

    // Batch-level gradient sums (reduced across threads after each batch)
    matrix_ptr H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
    matrix_ptr H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
    matrix_ptr L_W_grad_sum  = new_matrix(L_SIZE,  H1_SIZE);
    array_ptr  H0_B_grad_sum = new_array(H0_SIZE);
    array_ptr  H1_B_grad_sum = new_array(H1_SIZE);
    array_ptr  L_B_grad_sum  = new_array(L_SIZE);

    int *indices = malloc(TRAIN_SIZE * sizeof(int));
    for (int k = 0; k < TRAIN_SIZE; k++) indices[k] = k;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        // Fisher-Yates shuffle (skip first epoch to match serial baseline ordering)
        if (epoch > 0) {
            for (int k = TRAIN_SIZE - 1; k > 0; k--) {
                int r = rand() % (k + 1);
                int tmp = indices[k]; indices[k] = indices[r]; indices[r] = tmp;
            }
        }

        for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {

            // ── Parallel region: each thread handles one sub-batch ──────────────
            // Weights H0_W / H1_W / L_W and biases are read-only here.
            // Each thread writes only to its own private per-thread buffers,
            // so no locks or atomics are needed.
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();

                load_sub_batch(train_data, tid, indices, i);
                batched_feedforward(tid);
                batched_backprop(tid);
            }

            // ── Serial gradient reduction: sum per-thread results ───────────────
            // Each thread produced a full-sized weight gradient already summed
            // over its sub-batch samples (via GEMM).  We now sum across threads.
            for (int t = 0; t < num_threads; t++) {
                matrix_matrix_add(H0_W_grad_sum, H0_W_g_t[t], H0_W_grad_sum);
                matrix_matrix_add(H1_W_grad_sum, H1_W_g_t[t], H1_W_grad_sum);
                matrix_matrix_add(L_W_grad_sum,  L_W_g_t[t],  L_W_grad_sum);
                vector_vector_add(H0_B_grad_sum, H0_B_g_t[t], H0_B_grad_sum);
                vector_vector_add(H1_B_grad_sum, H1_B_g_t[t], H1_B_grad_sum);
                vector_vector_add(L_B_grad_sum,  L_B_g_t[t],  L_B_grad_sum);
            }

            // ── Parameter update (identical structure to serial baseline) ────────
            data_t reciprocal = 1.0f / BATCH_SIZE;

            matrix_scalar_mult(H0_W_grad_sum, reciprocal,          H0_W_grad_sum);
            matrix_scalar_mult(H0_W_grad_sum, (data_t)LEARN_RATE,  H0_W_grad_sum);
            matrix_scalar_mult(H0_W_grad_sum, -1.0f,               H0_W_grad_sum);

            matrix_scalar_mult(H1_W_grad_sum, reciprocal,          H1_W_grad_sum);
            matrix_scalar_mult(H1_W_grad_sum, (data_t)LEARN_RATE,  H1_W_grad_sum);
            matrix_scalar_mult(H1_W_grad_sum, -1.0f,               H1_W_grad_sum);

            matrix_scalar_mult(L_W_grad_sum,  reciprocal,          L_W_grad_sum);
            matrix_scalar_mult(L_W_grad_sum,  (data_t)LEARN_RATE,  L_W_grad_sum);
            matrix_scalar_mult(L_W_grad_sum,  -1.0f,               L_W_grad_sum);

            vector_scalar_mult(H0_B_grad_sum, reciprocal,          H0_B_grad_sum);
            vector_scalar_mult(H0_B_grad_sum, (data_t)LEARN_RATE,  H0_B_grad_sum);
            vector_scalar_mult(H0_B_grad_sum, -1.0f,               H0_B_grad_sum);

            vector_scalar_mult(H1_B_grad_sum, reciprocal,          H1_B_grad_sum);
            vector_scalar_mult(H1_B_grad_sum, (data_t)LEARN_RATE,  H1_B_grad_sum);
            vector_scalar_mult(H1_B_grad_sum, -1.0f,               H1_B_grad_sum);

            vector_scalar_mult(L_B_grad_sum,  reciprocal,          L_B_grad_sum);
            vector_scalar_mult(L_B_grad_sum,  (data_t)LEARN_RATE,  L_B_grad_sum);
            vector_scalar_mult(L_B_grad_sum,  -1.0f,               L_B_grad_sum);

            matrix_matrix_add(H0_W_grad_sum, H0_W, H0_W);
            matrix_matrix_add(H1_W_grad_sum, H1_W, H1_W);
            matrix_matrix_add(L_W_grad_sum,  L_W,  L_W);
            vector_vector_add(H0_B_grad_sum, H0_B, H0_B);
            vector_vector_add(H1_B_grad_sum, H1_B, H1_B);
            vector_vector_add(L_B_grad_sum,  L_B,  L_B);

            // Zero batch-level accumulators for next batch
            zero_matrix(H0_W_grad_sum);
            zero_matrix(H1_W_grad_sum);
            zero_matrix(L_W_grad_sum);
            zero_array(H0_B_grad_sum);
            zero_array(H1_B_grad_sum);
            zero_array(L_B_grad_sum);
        }
    }

    free(indices);
}


// ─── Test (serial, single-sample inference) ───────────────────────────────────
//
// Uses the trained shared weights.  Single-sample feedforward via matrix-vector
// multiply -- no need to batch inference.

/* Use the same fast sigmoid as training: sigma(x) = 0.5 + 0.5*x/(1+|x|) */
static data_t sig_infer(data_t z) { return 0.5f + 0.5f * z / (1.0f + fabsf(z)); }

static void test_parallel_MNIST(dataset_ptr test_data) {
    // Allocate small working buffers for single-sample inference
    array_ptr in    = new_array(I_SIZE);
    array_ptr h0_z  = new_array(H0_SIZE);
    array_ptr h0_a  = new_array(H0_SIZE);
    array_ptr h0_mv = new_array(H0_SIZE);
    array_ptr h1_z  = new_array(H1_SIZE);
    array_ptr h1_a  = new_array(H1_SIZE);
    array_ptr h1_mv = new_array(H1_SIZE);
    array_ptr out_z = new_array(L_SIZE);
    array_ptr out_a = new_array(L_SIZE);
    array_ptr l_mv  = new_array(L_SIZE);

    int correct = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        copyImageToInput(test_data, in, i);

        // Layer 0
        matrix_vector_mult(H0_W, in, h0_mv);
        vector_vector_add(H0_B, h0_mv, h0_z);
        for (int k = 0; k < H0_SIZE; k++) h0_a->data[k] = sig_infer(h0_z->data[k]);

        // Layer 1
        matrix_vector_mult(H1_W, h0_a, h1_mv);
        vector_vector_add(H1_B, h1_mv, h1_z);
        for (int k = 0; k < H1_SIZE; k++) h1_a->data[k] = sig_infer(h1_z->data[k]);

        // Output layer
        matrix_vector_mult(L_W, h1_a, l_mv);
        vector_vector_add(L_B, l_mv, out_z);
        for (int k = 0; k < L_SIZE; k++) out_a->data[k] = sig_infer(out_z->data[k]);

        output_max pred = vector_max(out_a);
        if (pred.index == test_data->nums[i]) correct++;
    }

    printf("Test accuracy: %d / %d (%.2f%%)\n", correct, TEST_SIZE,
           (float)correct / TEST_SIZE * 100.0f);
}


// ─── Public entry point ───────────────────────────────────────────────────────

void parallel_MNIST(dataset_ptr train_data, dataset_ptr test_data, int num_threads) {
    if (BATCH_SIZE % num_threads != 0) {
        fprintf(stderr, "Error: num_threads (%d) must evenly divide BATCH_SIZE (%d)\n",
                num_threads, BATCH_SIZE);
        exit(1);
    }

    printf("parallel_MNIST: %d threads, %d samples/thread\n",
           num_threads, BATCH_SIZE / num_threads);

    struct timespec t0, t1;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    init_parallel_MNIST(num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("init:  %.4f s\n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    train_parallel_MNIST(train_data, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("train: %.4f s\n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    test_parallel_MNIST(test_data);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("test:  %.4f s\n", (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9);
}
