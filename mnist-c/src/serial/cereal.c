#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "../params.h"
#include "array_matrix_funcs.h"
#include "array_matrix_math.h"
#include "kernels.h"
#include "cereal.h"

#define I_SIZE  IMAGE_SIZE
#define H0_SIZE 100 //100- model standardization
#define H1_SIZE 16  //16- model standardization
#define L_SIZE  10  //10- model standardization

#define BATCH_SIZE 100 //100- model standardization
#define LEARN_RATE 3   //3- model standardization
#define NUM_EPOCHS 10  //10- model standardization

#define NUM_THREADS 8  // explicit thread count for OpenMP parallel regions

// =====================================================================
// SHARED (read-only during parallel region): weights and biases
// These are written ONLY during the serial weight-update step
// =====================================================================
matrix_ptr H0_W; array_ptr H0_B;
matrix_ptr H1_W; array_ptr H1_B;
matrix_ptr L_W;  array_ptr L_B;

// =====================================================================
// PER-THREAD scratch: everything that gets written during a sample's
// forward+backward pass. One copy per thread (NOT per sample) -- a
// thread reuses its scratch sequentially across the samples it owns.
// =====================================================================
struct SampleScratch {
  // Activations + Z buffers
  array_ptr IN;
  array_ptr H0, H0_Z;
  array_ptr H1, H1_Z;
  array_ptr OUT, L_Z;

  // Per-sample gradients (overwritten every sample)
  matrix_ptr H0_W_grad, H1_W_grad, L_W_grad;
  array_ptr  H0_B_grad, H1_B_grad, L_B_grad;
  array_ptr  H0_A_grad, H1_A_grad;

  // Feedforward scratch
  array_ptr H0_mvm_res, H0_z_temp;
  array_ptr H1_mvm_res, H1_z_temp;
  array_ptr L_mvm_res,  L_z_temp;

  // Backprop scratch
  array_ptr  BP_delCdelA_L,  BP_y, BP_delAdelZ_L;
  matrix_ptr BP_W_T_L;
  array_ptr  BP_delCdelA_H1, BP_delAdelZ_H1;
  matrix_ptr BP_W_T_H1;
  array_ptr  BP_delCdelA_H0, BP_delAdelZ_H0;
};

// =====================================================================
// PER-THREAD gradient sums: each thread accumulates into its own slot
// during the parallel region. After the region, we serially combine
// these into the global sum used for the weight update.
// =====================================================================
typedef struct {
  matrix_ptr H0_W_grad_sum, H1_W_grad_sum, L_W_grad_sum;
  array_ptr  H0_B_grad_sum, H1_B_grad_sum, L_B_grad_sum;
} ThreadGradSum;

static SampleScratch  *scratch;       // size = NUM_THREADS
static ThreadGradSum  *thread_sums;   // size = NUM_THREADS

// Forward declarations -- updated signatures take a scratch pointer
void feedforward(SampleScratch *s);
int  backprop(SampleScratch *s, int num);
static void alloc_scratch(SampleScratch *s);
static void alloc_thread_sums(ThreadGradSum *t);
static void zero_thread_sums(ThreadGradSum *t);

// =====================================================================

void serial_MNIST(dataset_ptr train_data, dataset_ptr test_data) {
  struct timespec t0, t1;

  clock_gettime(CLOCK_MONOTONIC, &t0);
  init_MNIST();
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("init_MNIST:  %.4f s\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);

  clock_gettime(CLOCK_MONOTONIC, &t0);
  train_MNIST(train_data);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("train_MNIST: %.4f s\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);

  clock_gettime(CLOCK_MONOTONIC, &t0);
  test_MNIST(test_data);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  printf("test_MNIST:  %.4f s\n", (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
}

void init_MNIST() {
  omp_set_num_threads(NUM_THREADS);

  // SHARED weights and biases (read-only during parallel region)
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

  // PER-THREAD scratch + grad sums (allocate once for whole training run)
  scratch     = malloc(NUM_THREADS * sizeof(SampleScratch));
  thread_sums = malloc(NUM_THREADS * sizeof(ThreadGradSum));
  for (int t = 0; t < NUM_THREADS; t++) {
    alloc_scratch(&scratch[t]);
    alloc_thread_sums(&thread_sums[t]);
  }
}

static void alloc_scratch(SampleScratch *s) {
  s->IN   = new_array(I_SIZE);
  s->H0   = new_array(H0_SIZE);
  s->H0_Z = new_array(H0_SIZE);
  s->H1   = new_array(H1_SIZE);
  s->H1_Z = new_array(H1_SIZE);
  s->OUT  = new_array(L_SIZE);
  s->L_Z  = new_array(L_SIZE);

  s->H0_W_grad = new_matrix(H0_SIZE, I_SIZE);
  s->H1_W_grad = new_matrix(H1_SIZE, H0_SIZE);
  s->L_W_grad  = new_matrix(L_SIZE, H1_SIZE);
  s->H0_B_grad = new_array(H0_SIZE);
  s->H1_B_grad = new_array(H1_SIZE);
  s->L_B_grad  = new_array(L_SIZE);
  s->H0_A_grad = new_array(H0_SIZE);
  s->H1_A_grad = new_array(H1_SIZE);

  s->H0_mvm_res = new_array(H0_SIZE);
  s->H0_z_temp  = new_array(H0_SIZE);
  s->H1_mvm_res = new_array(H1_SIZE);
  s->H1_z_temp  = new_array(H1_SIZE);
  s->L_mvm_res  = new_array(L_SIZE);
  s->L_z_temp   = new_array(L_SIZE);

  s->BP_delCdelA_L  = new_array(L_SIZE);
  s->BP_y           = new_array(L_SIZE);
  s->BP_delAdelZ_L  = new_array(L_SIZE);
  s->BP_W_T_L       = new_matrix(H1_SIZE, L_SIZE);
  s->BP_delCdelA_H1 = new_array(H1_SIZE);
  s->BP_delAdelZ_H1 = new_array(H1_SIZE);
  s->BP_W_T_H1      = new_matrix(H0_SIZE, H1_SIZE);
  s->BP_delCdelA_H0 = new_array(H0_SIZE);
  s->BP_delAdelZ_H0 = new_array(H0_SIZE);
}

static void alloc_thread_sums(ThreadGradSum *t) {
  t->H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  t->H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  t->L_W_grad_sum  = new_matrix(L_SIZE, H1_SIZE);
  t->H0_B_grad_sum = new_array(H0_SIZE);
  t->H1_B_grad_sum = new_array(H1_SIZE);
  t->L_B_grad_sum  = new_array(L_SIZE);
  // new_matrix/new_array use calloc, so already zeroed
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

  // Global (final) gradient sum -- the destination after combining threads
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

      // ===== PARALLEL REGION: per-sample work =====
      // Each thread processes a chunk of samples, accumulating gradients
      // into its OWN thread_sums slot. No cross-thread writes.
      #pragma omp parallel for schedule(static)
      for (int j = i; j < i + BATCH_SIZE; j++) {
        int tid = omp_get_thread_num();
        SampleScratch *s  = &scratch[tid];
        ThreadGradSum *ts = &thread_sums[tid];

        copyImageToInput(train_data, s->IN, indices[j]);
        int num = train_data->nums[indices[j]];

        feedforward(s);
        backprop(s, num);

        // Accumulate this sample's gradients into THIS THREAD's sum.
        // Safe: no other thread touches ts.
        kernel_matrix_matrix_add(ts->H0_W_grad_sum, s->H0_W_grad, ts->H0_W_grad_sum);
        kernel_matrix_matrix_add(ts->H1_W_grad_sum, s->H1_W_grad, ts->H1_W_grad_sum);
        kernel_matrix_matrix_add(ts->L_W_grad_sum,  s->L_W_grad,  ts->L_W_grad_sum);
        vector_vector_add(ts->H0_B_grad_sum, s->H0_B_grad, ts->H0_B_grad_sum);
        vector_vector_add(ts->H1_B_grad_sum, s->H1_B_grad, ts->H1_B_grad_sum);
        vector_vector_add(ts->L_B_grad_sum,  s->L_B_grad,  ts->L_B_grad_sum);
      }
      // ===== END PARALLEL REGION (implicit barrier) =====

      // ===== SERIAL: combine thread-local sums into global sum =====
      for (int t = 0; t < NUM_THREADS; t++) {
        kernel_matrix_matrix_add(H0_W_grad_sum, thread_sums[t].H0_W_grad_sum, H0_W_grad_sum);
        kernel_matrix_matrix_add(H1_W_grad_sum, thread_sums[t].H1_W_grad_sum, H1_W_grad_sum);
        kernel_matrix_matrix_add(L_W_grad_sum,  thread_sums[t].L_W_grad_sum,  L_W_grad_sum);
        vector_vector_add(H0_B_grad_sum, thread_sums[t].H0_B_grad_sum, H0_B_grad_sum);
        vector_vector_add(H1_B_grad_sum, thread_sums[t].H1_B_grad_sum, H1_B_grad_sum);
        vector_vector_add(L_B_grad_sum,  thread_sums[t].L_B_grad_sum,  L_B_grad_sum);
        zero_thread_sums(&thread_sums[t]);  // reset for next batch
      }

      // ===== SERIAL: scale, negate, apply to weights/biases =====
      data_t reciprocalBatchSize = 1.0 / BATCH_SIZE;

      matrix_scalar_mult(H0_W_grad_sum, reciprocalBatchSize, H0_W_grad_sum);
      matrix_scalar_mult(H0_W_grad_sum, (data_t)LEARN_RATE,  H0_W_grad_sum);
      matrix_scalar_mult(H0_W_grad_sum, -1.0,                H0_W_grad_sum);

      matrix_scalar_mult(H1_W_grad_sum, reciprocalBatchSize, H1_W_grad_sum);
      matrix_scalar_mult(H1_W_grad_sum, (data_t)LEARN_RATE,  H1_W_grad_sum);
      matrix_scalar_mult(H1_W_grad_sum, -1.0,                H1_W_grad_sum);

      matrix_scalar_mult(L_W_grad_sum, reciprocalBatchSize, L_W_grad_sum);
      matrix_scalar_mult(L_W_grad_sum, (data_t)LEARN_RATE,  L_W_grad_sum);
      matrix_scalar_mult(L_W_grad_sum, -1.0,                L_W_grad_sum);

      vector_scalar_mult(H0_B_grad_sum, reciprocalBatchSize, H0_B_grad_sum);
      vector_scalar_mult(H0_B_grad_sum, (data_t)LEARN_RATE,  H0_B_grad_sum);
      vector_scalar_mult(H0_B_grad_sum, -1.0,                H0_B_grad_sum);

      vector_scalar_mult(H1_B_grad_sum, reciprocalBatchSize, H1_B_grad_sum);
      vector_scalar_mult(H1_B_grad_sum, (data_t)LEARN_RATE,  H1_B_grad_sum);
      vector_scalar_mult(H1_B_grad_sum, -1.0,                H1_B_grad_sum);

      vector_scalar_mult(L_B_grad_sum, reciprocalBatchSize, L_B_grad_sum);
      vector_scalar_mult(L_B_grad_sum, (data_t)LEARN_RATE,  L_B_grad_sum);
      vector_scalar_mult(L_B_grad_sum, -1.0,                L_B_grad_sum);

      kernel_matrix_matrix_add(H0_W_grad_sum, H0_W, H0_W);
      kernel_matrix_matrix_add(H1_W_grad_sum, H1_W, H1_W);
      kernel_matrix_matrix_add(L_W_grad_sum,  L_W,  L_W);
      vector_vector_add(H0_B_grad_sum, H0_B, H0_B);
      vector_vector_add(H1_B_grad_sum, H1_B, H1_B);
      vector_vector_add(L_B_grad_sum,  L_B,  L_B);

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
  // Test runs serially -- use thread 0's scratch as a convenient buffer.
  SampleScratch *s = &scratch[0];
  int correct = 0;

  for (int i = 0; i < TEST_SIZE; i++) {
    copyImageToInput(test_data, s->IN, i);
    feedforward(s);

    output_max prediction = vector_max(s->OUT);
    int label = test_data->nums[i];
    if (prediction.index == label) correct++;
  }

  data_t accuracy = (data_t)correct / TEST_SIZE;
  printf("Test accuracy: %d / %d (%.2f%%)\n", correct, TEST_SIZE, accuracy * 100);
}

// =====================================================================

int backprop(SampleScratch *s, int num) {
  // OUTPUT LAYER
  zero_array(s->BP_y);
  s->BP_y->data[num] = 1;
  if (!vector_vector_sub(s->OUT, s->BP_y, s->BP_delCdelA_L)) return 0;

  if (!vector_copy(s->L_Z, s->BP_delAdelZ_L)) return 0;
  sigmoid_prime_arr(s->BP_delAdelZ_L);

  if (!vector_vector_elementwise_mult(s->BP_delAdelZ_L, s->BP_delCdelA_L, s->L_B_grad)) return 0;
  if (!kernel_vector_vector_mult(s->L_B_grad, s->H1, s->L_W_grad)) return 0;

  if (!matrix_transpose(L_W, s->BP_W_T_L)) return 0;       // L_W is SHARED (read-only)
  if (!kernel_matrix_vector_mult(s->BP_W_T_L, s->L_B_grad, s->H1_A_grad)) return 0;

  // HIDDEN LAYER 1
  if (!vector_copy(s->H1_A_grad, s->BP_delCdelA_H1)) return 0;
  if (!vector_copy(s->H1_Z, s->BP_delAdelZ_H1)) return 0;
  sigmoid_prime_arr(s->BP_delAdelZ_H1);

  if (!vector_vector_elementwise_mult(s->BP_delAdelZ_H1, s->BP_delCdelA_H1, s->H1_B_grad)) return 0;
  if (!kernel_vector_vector_mult(s->H1_B_grad, s->H0, s->H1_W_grad)) return 0;

  if (!matrix_transpose(H1_W, s->BP_W_T_H1)) return 0;     // H1_W is SHARED (read-only)
  if (!kernel_matrix_vector_mult(s->BP_W_T_H1, s->H1_B_grad, s->H0_A_grad)) return 0;

  // HIDDEN LAYER 0
  if (!vector_copy(s->H0_A_grad, s->BP_delCdelA_H0)) return 0;
  if (!vector_copy(s->H0_Z, s->BP_delAdelZ_H0)) return 0;
  sigmoid_prime_arr(s->BP_delAdelZ_H0);

  if (!vector_vector_elementwise_mult(s->BP_delAdelZ_H0, s->BP_delCdelA_H0, s->H0_B_grad)) return 0;
  if (!kernel_vector_vector_mult(s->H0_B_grad, s->IN, s->H0_W_grad)) return 0;

  return 1;
}

void feedforward(SampleScratch *s) {
  // IN -> H0    (H0_W and H0_B are SHARED, read-only here)
  kernel_matrix_vector_mult(H0_W, s->IN, s->H0_mvm_res);
  vector_vector_add(H0_B, s->H0_mvm_res, s->H0_z_temp);
  vector_copy(s->H0_z_temp, s->H0_Z);
  sigmoid_arr(s->H0_z_temp);
  vector_copy(s->H0_z_temp, s->H0);

  // H0 -> H1
  kernel_matrix_vector_mult(H1_W, s->H0, s->H1_mvm_res);
  vector_vector_add(H1_B, s->H1_mvm_res, s->H1_z_temp);
  vector_copy(s->H1_z_temp, s->H1_Z);
  sigmoid_arr(s->H1_z_temp);
  vector_copy(s->H1_z_temp, s->H1);

  // H1 -> OUT
  kernel_matrix_vector_mult(L_W, s->H1, s->L_mvm_res);
  vector_vector_add(L_B, s->L_mvm_res, s->L_z_temp);
  vector_copy(s->L_z_temp, s->L_Z);
  sigmoid_arr(s->L_z_temp);
  vector_copy(s->L_z_temp, s->OUT);
}

void sigmoid_arr(array_ptr v) {
  for (int i = 0; i < v->len; i++) v->data[i] = sigmoid(v->data[i]);
}

void sigmoid_prime_arr(array_ptr v) {
  for (int i = 0; i < v->len; i++) v->data[i] = sigmoid_prime(v->data[i]);
}

data_t sigmoid(data_t z)       { return 1 / (1 + expf(-z)); }
data_t sigmoid_prime(data_t z) { return sigmoid(z) * (1 - sigmoid(z)); }

array_ptr numToVec(int num) {
  array_ptr result = new_array(L_SIZE);
  result->data[num] = 1;
  return result;
}