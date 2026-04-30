#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

#include "array_matrix_funcs.h"
#include "array_matrix_math.h"
#include "kernels.h"
#include "model.h"

#define I_SIZE  IMAGE_SIZE
#define H0_SIZE 100
#define H1_SIZE 16
#define L_SIZE  10

#define BATCH_SIZE  100
#define LEARN_RATE  3
#define NUM_EPOCHS  10
#define NUM_THREADS 8

/* max samples any single thread will ever own in one batch */
#define MAX_S ((BATCH_SIZE + NUM_THREADS - 1) / NUM_THREADS)

// =====================================================================
// SHARED (read-only during parallel region): weights and biases
// =====================================================================
matrix_ptr H0_W; array_ptr H0_B;
matrix_ptr H1_W; array_ptr H1_B;
matrix_ptr L_W;  array_ptr L_B;

// =====================================================================
// PER-THREAD batch scratch.
// All activation and gradient buffers are now matrices of shape
// (MAX_S x layer_size), storing one row per sample (samples-as-rows).
// =====================================================================
struct BatchScratch {
  /* Input batch: MAX_S x I_SIZE */
  matrix_ptr IN_batch;

  /* Activations and pre-activations: MAX_S x layer_size */
  matrix_ptr H0_batch,  H0_Z_batch;
  matrix_ptr H1_batch,  H1_Z_batch;
  matrix_ptr OUT_batch, L_Z_batch;

  /* Deltas (bias-gradient batch): MAX_S x layer_size */
  matrix_ptr L_delta_batch;
  matrix_ptr H1_delta_batch;
  matrix_ptr H0_delta_batch;

  /* Backprop error signals: MAX_S x layer_size */
  matrix_ptr H1_A_grad_batch;
  matrix_ptr H0_A_grad_batch;

  /* Feedforward intermediates: MAX_S x layer_size */
  matrix_ptr H0_mm_res, H0_z_temp;
  matrix_ptr H1_mm_res, H1_z_temp;
  matrix_ptr L_mm_res,  L_z_temp;

  /* Backprop intermediates */
  matrix_ptr BP_delCdelA_L_batch;   /* MAX_S x L_SIZE  */
  matrix_ptr BP_delAdelZ_L_batch;   /* MAX_S x L_SIZE  */
  matrix_ptr BP_W_T_L;              /* H1_SIZE x L_SIZE  (L_W transposed) */
  matrix_ptr BP_delAdelZ_H1_batch;  /* MAX_S x H1_SIZE */
  matrix_ptr BP_W_T_H1;             /* H0_SIZE x H1_SIZE (H1_W transposed) */
  matrix_ptr BP_delAdelZ_H0_batch;  /* MAX_S x H0_SIZE */

  /* Label scratch */
  int *labels;  /* MAX_S ints */
};

static BatchScratch *scratch;      /* size = NUM_THREADS */
static ThreadGradSum *thread_sums; /* size = NUM_THREADS */

/* Forward declarations */
static void alloc_scratch(BatchScratch *s);
static void alloc_thread_sums(ThreadGradSum *t);
static void zero_thread_sums(ThreadGradSum *t);

// =====================================================================

void parallel_MNIST(dataset_ptr train_data, dataset_ptr test_data) {
  struct timespec t0, t1;

  clock_gettime(CLOCK_MONOTONIC, &t0);
  init_MNIST();
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("init_MNIST:  %.4f s\n",
         (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);

  clock_gettime(CLOCK_MONOTONIC, &t0);
  train_MNIST(train_data);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("train_MNIST: %.4f s\n",
         (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);

  clock_gettime(CLOCK_MONOTONIC, &t0);
  test_MNIST(test_data);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("test_MNIST:  %.4f s\n",
         (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
}

void init_MNIST() {
  omp_set_num_threads(NUM_THREADS);

  H0_W = new_matrix(H0_SIZE, I_SIZE);
  H0_B = new_array(H0_SIZE);
  H1_W = new_matrix(H1_SIZE, H0_SIZE);
  H1_B = new_array(H1_SIZE);
  L_W  = new_matrix(L_SIZE, H1_SIZE);
  L_B  = new_array(L_SIZE);

  float rand_range = sqrt(1.0 / I_SIZE);
  init_matrix_rand(H0_W, -rand_range, rand_range);
  rand_range = sqrt(1.0 / H0_SIZE);
  init_matrix_rand(H1_W, -rand_range, rand_range);
  rand_range = sqrt(1.0 / H1_SIZE);
  init_matrix_rand(L_W, -rand_range, rand_range);

  scratch     = malloc(NUM_THREADS * sizeof(BatchScratch));
  thread_sums = malloc(NUM_THREADS * sizeof(ThreadGradSum));
  for (int t = 0; t < NUM_THREADS; t++) {
    alloc_scratch(&scratch[t]);
    alloc_thread_sums(&thread_sums[t]);
  }
}

static void alloc_scratch(BatchScratch *s) {
  s->IN_batch  = new_matrix(MAX_S, I_SIZE);

  s->H0_batch  = new_matrix(MAX_S, H0_SIZE);
  s->H0_Z_batch = new_matrix(MAX_S, H0_SIZE);
  s->H1_batch  = new_matrix(MAX_S, H1_SIZE);
  s->H1_Z_batch = new_matrix(MAX_S, H1_SIZE);
  s->OUT_batch = new_matrix(MAX_S, L_SIZE);
  s->L_Z_batch = new_matrix(MAX_S, L_SIZE);

  s->L_delta_batch  = new_matrix(MAX_S, L_SIZE);
  s->H1_delta_batch = new_matrix(MAX_S, H1_SIZE);
  s->H0_delta_batch = new_matrix(MAX_S, H0_SIZE);

  s->H1_A_grad_batch = new_matrix(MAX_S, H1_SIZE);
  s->H0_A_grad_batch = new_matrix(MAX_S, H0_SIZE);

  s->H0_mm_res = new_matrix(MAX_S, H0_SIZE);
  s->H0_z_temp = new_matrix(MAX_S, H0_SIZE);
  s->H1_mm_res = new_matrix(MAX_S, H1_SIZE);
  s->H1_z_temp = new_matrix(MAX_S, H1_SIZE);
  s->L_mm_res  = new_matrix(MAX_S, L_SIZE);
  s->L_z_temp  = new_matrix(MAX_S, L_SIZE);

  s->BP_delCdelA_L_batch  = new_matrix(MAX_S, L_SIZE);
  s->BP_delAdelZ_L_batch  = new_matrix(MAX_S, L_SIZE);
  s->BP_W_T_L             = new_matrix(H1_SIZE, L_SIZE);
  s->BP_delAdelZ_H1_batch = new_matrix(MAX_S, H1_SIZE);
  s->BP_W_T_H1            = new_matrix(H0_SIZE, H1_SIZE);
  s->BP_delAdelZ_H0_batch = new_matrix(MAX_S, H0_SIZE);

  s->labels = malloc(MAX_S * sizeof(int));
}

static void alloc_thread_sums(ThreadGradSum *t) {
  t->H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  t->H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  t->L_W_grad_sum  = new_matrix(L_SIZE, H1_SIZE);
  t->H0_B_grad_sum = new_array(H0_SIZE);
  t->H1_B_grad_sum = new_array(H1_SIZE);
  t->L_B_grad_sum  = new_array(L_SIZE);
  /* new_matrix / new_array use calloc, already zeroed */
}

static void zero_thread_sums(ThreadGradSum *t) {
  zero_matrix(t->H0_W_grad_sum);
  zero_matrix(t->H1_W_grad_sum);
  zero_matrix(t->L_W_grad_sum);
  zero_array(t->H0_B_grad_sum);
  zero_array(t->H1_B_grad_sum);
  zero_array(t->L_B_grad_sum);
}

// =====================================================================

void train_MNIST(dataset_ptr train_data) {

  matrix_ptr H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  matrix_ptr H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  matrix_ptr L_W_grad_sum  = new_matrix(L_SIZE, H1_SIZE);
  array_ptr  H0_B_grad_sum = new_array(H0_SIZE);
  array_ptr  H1_B_grad_sum = new_array(H1_SIZE);
  array_ptr  L_B_grad_sum  = new_array(L_SIZE);

  int *indices = malloc(TRAIN_SIZE * sizeof(int));
  for (int k = 0; k < TRAIN_SIZE; k++) indices[k] = k;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

    if (epoch > 0) {
      for (int k = TRAIN_SIZE - 1; k > 0; k--) {
        int r = rand() % (k + 1);
        int tmp = indices[k];
        indices[k] = indices[r];
        indices[r] = tmp;
      }
    }

    for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {

      /* ===== PARALLEL REGION =====
       * Each thread computes its own contiguous sub-batch of samples,
       * packs them into a batch matrix, runs feedforward + backprop
       * on all of them at once, and accumulates gradients into its
       * private ThreadGradSum.  No per-sample loop within the thread.
       */
      #pragma omp parallel
      {
        int tid      = omp_get_thread_num();
        int start    = (tid * BATCH_SIZE) / NUM_THREADS;
        int end      = ((tid + 1) * BATCH_SIZE) / NUM_THREADS;
        int actual_S = end - start;

        BatchScratch  *s  = &scratch[tid];
        ThreadGradSum *ts = &thread_sums[tid];

        /* Pack images: indices[i+start .. i+start+actual_S-1] → IN_batch */
        copyImagesToInputBatch(train_data, s->IN_batch,
                               &indices[i + start], actual_S);

        /* Pack labels */
        for (int k = 0; k < actual_S; k++)
          s->labels[k] = train_data->nums[indices[i + start + k]];

        feedforward_batch(s, actual_S);
        backprop_batch(s, ts, actual_S);
      }
      /* ===== END PARALLEL REGION (implicit barrier) ===== */

      /* ===== SERIAL: combine per-thread sums into global sum ===== */
      for (int t = 0; t < NUM_THREADS; t++) {
        kernel_matrix_matrix_add(H0_W_grad_sum,
                                 thread_sums[t].H0_W_grad_sum, H0_W_grad_sum);
        kernel_matrix_matrix_add(H1_W_grad_sum,
                                 thread_sums[t].H1_W_grad_sum, H1_W_grad_sum);
        kernel_matrix_matrix_add(L_W_grad_sum,
                                 thread_sums[t].L_W_grad_sum,  L_W_grad_sum);
        vector_vector_add(H0_B_grad_sum,
                                 thread_sums[t].H0_B_grad_sum, H0_B_grad_sum);
        vector_vector_add(H1_B_grad_sum,
                                 thread_sums[t].H1_B_grad_sum, H1_B_grad_sum);
        vector_vector_add(L_B_grad_sum,
                                 thread_sums[t].L_B_grad_sum,  L_B_grad_sum);
        zero_thread_sums(&thread_sums[t]);
      }

      /* ===== SERIAL: scale and apply to weights/biases ===== */
      data_t scale = -(data_t)LEARN_RATE / BATCH_SIZE;

      kernel_matrix_saxpy(H0_W_grad_sum, scale, H0_W);
      kernel_matrix_saxpy(H1_W_grad_sum, scale, H1_W);
      kernel_matrix_saxpy(L_W_grad_sum,  scale, L_W);

      kernel_vector_saxpy(H0_B_grad_sum, scale, H0_B);
      kernel_vector_saxpy(H1_B_grad_sum, scale, H1_B);
      kernel_vector_saxpy(L_B_grad_sum,  scale, L_B);

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

// =====================================================================

void test_MNIST(dataset_ptr test_data) {
  /* Test runs serially using a single-sample feedforward path.
   * We re-use scratch[0]'s IN_batch (row 0) as the input vector wrapper,
   * but the single-sample kernel_matrix_vector_mult path is simpler.
   * We build a one-row batch (actual_S=1) and call feedforward_batch. */
  BatchScratch *s = &scratch[0];
  int correct = 0;

  for (int i = 0; i < TEST_SIZE; i++) {
    /* Load one image into row 0 of IN_batch */
    data_t *dst = get_matrix_start(s->IN_batch);
    data_t *src = &test_data->image_arr[(long int)i * I_SIZE];
    for (int j = 0; j < I_SIZE; j++) dst[j] = src[j];

    feedforward_batch(s, 1);

    /* Find argmax in OUT_batch row 0 */
    data_t *out = get_matrix_start(s->OUT_batch);
    int    pred = 0;
    data_t best = out[0];
    for (int j = 1; j < L_SIZE; j++) {
      if (out[j] > best) { best = out[j]; pred = j; }
    }

    if (pred == test_data->nums[i]) correct++;
  }

  data_t accuracy = (data_t)correct / TEST_SIZE;
  printf("Test accuracy: %d / %d (%.2f%%)\n",
         correct, TEST_SIZE, accuracy * 100);
}

// =====================================================================

void feedforward_batch(BatchScratch *s, int actual_S) {

  /* IN_batch → H0
   * H0_mm_res = IN_batch x H0_W^T   (actual_S x H0_SIZE) */
  kernel_gemm_forward(s->IN_batch, H0_W, s->H0_mm_res, actual_S);
  kernel_bias_broadcast_add(s->H0_mm_res, H0_B, actual_S);
  kernel_matrix_copy_rows(s->H0_mm_res, s->H0_Z_batch, actual_S); /* save Z */
  sigmoid_mat(s->H0_mm_res, actual_S);
  kernel_matrix_copy_rows(s->H0_mm_res, s->H0_batch, actual_S);   /* save A */

  /* H0_batch → H1 */
  kernel_gemm_forward(s->H0_batch, H1_W, s->H1_mm_res, actual_S);
  kernel_bias_broadcast_add(s->H1_mm_res, H1_B, actual_S);
  kernel_matrix_copy_rows(s->H1_mm_res, s->H1_Z_batch, actual_S);
  sigmoid_mat(s->H1_mm_res, actual_S);
  kernel_matrix_copy_rows(s->H1_mm_res, s->H1_batch, actual_S);

  /* H1_batch → OUT */
  kernel_gemm_forward(s->H1_batch, L_W, s->L_mm_res, actual_S);
  kernel_bias_broadcast_add(s->L_mm_res, L_B, actual_S);
  kernel_matrix_copy_rows(s->L_mm_res, s->L_Z_batch, actual_S);
  sigmoid_mat(s->L_mm_res, actual_S);
  kernel_matrix_copy_rows(s->L_mm_res, s->OUT_batch, actual_S);
}

// =====================================================================

void backprop_batch(BatchScratch *s, ThreadGradSum *ts, int actual_S) {

  long int cols_L  = L_SIZE;
  long int cols_H1 = H1_SIZE;
  long int cols_H0 = H0_SIZE;
  long int cols_IN = I_SIZE;

  /* ------------------------------------------------------------------
   * OUTPUT LAYER
   * ------------------------------------------------------------------ */

  /* delCdelA_L = OUT_batch - e_num  (copy OUT, subtract 1 at label col) */
  kernel_matrix_copy_rows(s->OUT_batch, s->BP_delCdelA_L_batch, actual_S);
  {
    data_t *dc = get_matrix_start(s->BP_delCdelA_L_batch);
    for (int k = 0; k < actual_S; k++)
      dc[(long int)k * cols_L + s->labels[k]] -= 1.0f;
  }

  /* delAdelZ_L = sigmoid'(L_Z_batch) */
  kernel_matrix_copy_rows(s->L_Z_batch, s->BP_delAdelZ_L_batch, actual_S);
  sigmoid_prime_mat(s->BP_delAdelZ_L_batch, actual_S);

  /* delta_L = sigmoid'(L_Z) ⊙ (OUT - e_num) */
  kernel_hadamard_mat(s->BP_delAdelZ_L_batch, s->BP_delCdelA_L_batch,
                      s->L_delta_batch, actual_S);

  /* Accumulate L bias gradient */
  kernel_bias_grad_accum(s->L_delta_batch, ts->L_B_grad_sum, actual_S);

  /* Accumulate L weight gradient: dW += delta_L^T x H1_batch */
  kernel_gemm_weight_grad(s->L_delta_batch, s->H1_batch,
                          ts->L_W_grad_sum, actual_S);

  /* Propagate error to H1: H1_A_grad = delta_L x L_W
   * Using kernel_gemm_forward(delta_L, L_W_T, H1_A_grad) where
   * L_W_T = L_W^T  (H1_SIZE x L_SIZE), so B^T = L_W. */
  matrix_transpose(L_W, s->BP_W_T_L);  /* computed once per batch */
  kernel_gemm_forward(s->L_delta_batch, s->BP_W_T_L,
                      s->H1_A_grad_batch, actual_S);

  /* ------------------------------------------------------------------
   * HIDDEN LAYER 1
   * ------------------------------------------------------------------ */

  /* delAdelZ_H1 = sigmoid'(H1_Z_batch) */
  kernel_matrix_copy_rows(s->H1_Z_batch, s->BP_delAdelZ_H1_batch, actual_S);
  sigmoid_prime_mat(s->BP_delAdelZ_H1_batch, actual_S);

  /* delta_H1 = sigmoid'(H1_Z) ⊙ H1_A_grad */
  kernel_hadamard_mat(s->BP_delAdelZ_H1_batch, s->H1_A_grad_batch,
                      s->H1_delta_batch, actual_S);

  /* Accumulate H1 bias gradient */
  kernel_bias_grad_accum(s->H1_delta_batch, ts->H1_B_grad_sum, actual_S);

  /* Accumulate H1 weight gradient: dW += delta_H1^T x H0_batch */
  kernel_gemm_weight_grad(s->H1_delta_batch, s->H0_batch,
                          ts->H1_W_grad_sum, actual_S);

  /* Propagate error to H0: H0_A_grad = delta_H1 x H1_W */
  matrix_transpose(H1_W, s->BP_W_T_H1);
  kernel_gemm_forward(s->H1_delta_batch, s->BP_W_T_H1,
                      s->H0_A_grad_batch, actual_S);

  /* ------------------------------------------------------------------
   * HIDDEN LAYER 0
   * ------------------------------------------------------------------ */

  /* delAdelZ_H0 = sigmoid'(H0_Z_batch) */
  kernel_matrix_copy_rows(s->H0_Z_batch, s->BP_delAdelZ_H0_batch, actual_S);
  sigmoid_prime_mat(s->BP_delAdelZ_H0_batch, actual_S);

  /* delta_H0 = sigmoid'(H0_Z) ⊙ H0_A_grad */
  kernel_hadamard_mat(s->BP_delAdelZ_H0_batch, s->H0_A_grad_batch,
                      s->H0_delta_batch, actual_S);

  /* Accumulate H0 bias gradient */
  kernel_bias_grad_accum(s->H0_delta_batch, ts->H0_B_grad_sum, actual_S);

  /* Accumulate H0 weight gradient: dW += delta_H0^T x IN_batch
   * This is the dominant 100x784 hot loop. */
  kernel_gemm_weight_grad(s->H0_delta_batch, s->IN_batch,
                          ts->H0_W_grad_sum, actual_S);
}
