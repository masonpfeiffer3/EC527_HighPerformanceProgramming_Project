#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../params.h"
#include "array_matrix_funcs.h"
#include "array_matrix_math.h"
#include "kernels.h"
#include "cereal.h"

#define I_SIZE  IMAGE_SIZE
#define H0_SIZE 16 //increase model size with this
#define H1_SIZE 16 //increase model size with this
#define L_SIZE  10 //outputs

#define BATCH_SIZE 10 //experiment with this
#define LEARN_RATE 1 //expeirment with this
#define NUM_EPOCHS 2 //experiment with this

// Input layer
array_ptr IN;

// Hidden layer 0: activations, weights, biases, pre-activation (Z), gradients
array_ptr  H0;
matrix_ptr H0_W;
array_ptr  H0_B;
array_ptr  H0_Z;
matrix_ptr H0_W_grad;
array_ptr  H0_B_grad;

// Hidden layer 1
array_ptr  H1;
matrix_ptr H1_W;
array_ptr  H1_B;
array_ptr  H1_Z;
matrix_ptr H1_W_grad;
array_ptr  H1_B_grad;
array_ptr  H0_A_grad;

// Output layer
array_ptr  OUT;
matrix_ptr L_W;
array_ptr  L_B;
array_ptr  L_Z;
matrix_ptr L_W_grad;
array_ptr  L_B_grad;
array_ptr  H1_A_grad;

// Feedforward scratch buffers -- moved here from feedforward() to avoid per-call malloc (was leaking ~30MB/epoch)
array_ptr H0_mvm_res, H0_z_temp;
array_ptr H1_mvm_res, H1_z_temp;
array_ptr  L_mvm_res,  L_z_temp;

// Backprop scratch buffers -- moved here from backprop() to avoid per-call malloc (was leaking ~132MB/epoch)
array_ptr  BP_delCdelA_L, BP_y, BP_delAdelZ_L;
matrix_ptr BP_W_T_L;
array_ptr  BP_delCdelA_H1, BP_delAdelZ_H1;
matrix_ptr BP_W_T_H1;
array_ptr  BP_delCdelA_H0, BP_delAdelZ_H0;

void serial_MNIST(dataset_ptr train_data, dataset_ptr test_data) {
  init_MNIST();
  train_MNIST(train_data);
  test_MNIST(test_data);
}

void init_MNIST(){
  // STAGE 1: INITIALIZE LAYERS, WEIGHTS, BIASES, AND GRADIENT BUFFERS
  // Activations, Z buffers, and gradients don't need init -- they're
  // overwritten every pass. Biases stay at calloc's zero default.
  // Weights use Xavier-style ranges: +/- sqrt(1 / fan_in).

  IN = new_array(I_SIZE); //all zeros

  // Hidden layer 0
  H0        = new_array(H0_SIZE);
  H0_W      = new_matrix(H0_SIZE, I_SIZE);
  H0_B      = new_array(H0_SIZE);
  H0_Z      = new_array(H0_SIZE);
  H0_W_grad = new_matrix(H0_SIZE, I_SIZE);
  H0_B_grad = new_array(H0_SIZE);

  float rand_range = sqrt(1.0 / I_SIZE);
  init_matrix_rand(H0_W, -rand_range, rand_range);

  // Hidden layer 1
  H1        = new_array(H1_SIZE);
  H1_W      = new_matrix(H1_SIZE, H0_SIZE);
  H1_B      = new_array(H1_SIZE);
  H1_Z      = new_array(H1_SIZE);
  H1_W_grad = new_matrix(H1_SIZE, H0_SIZE);
  H1_B_grad = new_array(H1_SIZE);
  H0_A_grad = new_array(H0_SIZE);

  rand_range = sqrt(1.0 / H0_SIZE);
  init_matrix_rand(H1_W, -rand_range, rand_range);

  // Output layer
  OUT       = new_array(L_SIZE);
  L_W       = new_matrix(L_SIZE, H1_SIZE);
  L_B       = new_array(L_SIZE);
  L_Z       = new_array(L_SIZE);
  L_W_grad  = new_matrix(L_SIZE, H1_SIZE);
  L_B_grad  = new_array(L_SIZE);
  H1_A_grad = new_array(H1_SIZE);

  rand_range = sqrt(1.0 / H1_SIZE);
  init_matrix_rand(L_W, -rand_range, rand_range);

  // Feedforward scratch buffers
  H0_mvm_res = new_array(H0_SIZE);
  H0_z_temp  = new_array(H0_SIZE);
  H1_mvm_res = new_array(H1_SIZE);
  H1_z_temp  = new_array(H1_SIZE);
  L_mvm_res  = new_array(L_SIZE);
  L_z_temp   = new_array(L_SIZE);

  // Backprop scratch buffers
  BP_delCdelA_L  = new_array(L_SIZE);
  BP_y           = new_array(L_SIZE);
  BP_delAdelZ_L  = new_array(L_SIZE);
  BP_W_T_L       = new_matrix(H1_SIZE, L_SIZE);
  BP_delCdelA_H1 = new_array(H1_SIZE);
  BP_delAdelZ_H1 = new_array(H1_SIZE);
  BP_W_T_H1      = new_matrix(H0_SIZE, H1_SIZE);
  BP_delCdelA_H0 = new_array(H0_SIZE);
  BP_delAdelZ_H0 = new_array(H0_SIZE);
}

void train_MNIST(dataset_ptr train_data){

  matrix_ptr H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  matrix_ptr H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  matrix_ptr L_W_grad_sum  = new_matrix(L_SIZE, H1_SIZE);

  array_ptr H0_B_grad_sum = new_array(H0_SIZE);
  array_ptr H1_B_grad_sum = new_array(H1_SIZE);
  array_ptr L_B_grad_sum  = new_array(L_SIZE);

  int *indices = malloc(TRAIN_SIZE * sizeof(int));
  for (int k = 0; k < TRAIN_SIZE; k++) indices[k] = k;

  for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

    // Fisher-Yates shuffle (skip on first epoch to preserve original ordering)
    if (epoch > 0) {
      for (int k = TRAIN_SIZE - 1; k > 0; k--) {
        int r = rand() % (k + 1);
        int tmp = indices[k];
        indices[k] = indices[r];
        indices[r] = tmp;
      }
    }

    for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {

      for (int j = i; j < i + BATCH_SIZE; j++) {
        copyImageToInput(train_data, IN, indices[j]);
        int num = train_data->nums[indices[j]];

        feedforward();

        if (!backprop(num)) printf("backprop failed!");

        // HOTSPOT: H0_W_grad_sum += H0_W_grad   [16x784] + [16x784] -> [16x784]
        // R: 25,088 (2 x 12,544)   W: 12,544   FLOPs: 12,544 (add)
        kernel_matrix_matrix_add(H0_W_grad_sum, H0_W_grad, H0_W_grad_sum);
        matrix_matrix_add(H1_W_grad_sum, H1_W_grad, H1_W_grad_sum);
        matrix_matrix_add(L_W_grad_sum,  L_W_grad,  L_W_grad_sum);

        vector_vector_add(H0_B_grad_sum, H0_B_grad, H0_B_grad_sum);
        vector_vector_add(H1_B_grad_sum, H1_B_grad, H1_B_grad_sum);
        vector_vector_add(L_B_grad_sum,  L_B_grad,  L_B_grad_sum);
      }

      data_t reciprocalBatchSize = 1.0 / BATCH_SIZE;
      // Average weight gradients, scale by learning rate, negate
      matrix_scalar_mult(H0_W_grad_sum, reciprocalBatchSize, H0_W_grad_sum);
      matrix_scalar_mult(H0_W_grad_sum, (data_t)LEARN_RATE,  H0_W_grad_sum);
      matrix_scalar_mult(H0_W_grad_sum, -1.0,                H0_W_grad_sum);

      matrix_scalar_mult(H1_W_grad_sum, reciprocalBatchSize, H1_W_grad_sum);
      matrix_scalar_mult(H1_W_grad_sum, (data_t)LEARN_RATE,  H1_W_grad_sum);
      matrix_scalar_mult(H1_W_grad_sum, -1.0,                H1_W_grad_sum);

      matrix_scalar_mult(L_W_grad_sum, reciprocalBatchSize, L_W_grad_sum);
      matrix_scalar_mult(L_W_grad_sum, (data_t)LEARN_RATE,  L_W_grad_sum);
      matrix_scalar_mult(L_W_grad_sum, -1.0,                L_W_grad_sum);
      // Average bias gradients, scale by learning rate, negate
      vector_scalar_mult(H0_B_grad_sum, reciprocalBatchSize, H0_B_grad_sum);
      vector_scalar_mult(H0_B_grad_sum, (data_t)LEARN_RATE,  H0_B_grad_sum);
      vector_scalar_mult(H0_B_grad_sum, -1.0,                H0_B_grad_sum);

      vector_scalar_mult(H1_B_grad_sum, reciprocalBatchSize, H1_B_grad_sum);
      vector_scalar_mult(H1_B_grad_sum, (data_t)LEARN_RATE,  H1_B_grad_sum);
      vector_scalar_mult(H1_B_grad_sum, -1.0,                H1_B_grad_sum);

      vector_scalar_mult(L_B_grad_sum, reciprocalBatchSize, L_B_grad_sum);
      vector_scalar_mult(L_B_grad_sum, (data_t)LEARN_RATE,  L_B_grad_sum);
      vector_scalar_mult(L_B_grad_sum, -1.0,                L_B_grad_sum);
      // Apply updates to weights and biases
      matrix_matrix_add(H0_W_grad_sum, H0_W, H0_W);
      matrix_matrix_add(H1_W_grad_sum, H1_W, H1_W);
      matrix_matrix_add(L_W_grad_sum,  L_W,  L_W);

      vector_vector_add(H0_B_grad_sum, H0_B, H0_B);
      vector_vector_add(H1_B_grad_sum, H1_B, H1_B);
      vector_vector_add(L_B_grad_sum,  L_B,  L_B);
      // Zero out accumulators for the next batch
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

void test_MNIST(dataset_ptr test_data) {
  int correct = 0;

  for (int i = 0; i < TEST_SIZE; i++) {
    // plug in each of the tests into the parameter set
    copyImageToInput(test_data, IN, i);
    feedforward();

    // check the output, use the max function
    output_max prediction = vector_max(OUT);
    int label = test_data->nums[i];

    // if the index matches the label, accumulate a correct
    if (prediction.index == label) {
      correct++;
    } 
  }

  // determine accuracy
  data_t accuracy = (data_t)correct / TEST_SIZE;
  printf("Test accuracy: %d / %d (%.2f%%)\n", correct, TEST_SIZE, accuracy * 100);
}


int backprop(int num) {

  // OUTPUT LAYER
  zero_array(BP_y);
  BP_y->data[num] = 1;
  if (!vector_vector_sub(OUT, BP_y, BP_delCdelA_L)) return 0;

  if (!vector_copy(L_Z, BP_delAdelZ_L)) return 0;
  sigmoid_prime_arr(BP_delAdelZ_L);

  if (!vector_vector_elementwise_mult(BP_delAdelZ_L, BP_delCdelA_L, L_B_grad)) return 0;
  if (!vector_vector_mult(L_B_grad, H1, L_W_grad)) return 0;

  if (!matrix_transpose(L_W, BP_W_T_L)) return 0;
  if (!matrix_vector_mult(BP_W_T_L, L_B_grad, H1_A_grad)) return 0;

  // HIDDEN LAYER 1
  if (!vector_copy(H1_A_grad, BP_delCdelA_H1)) return 0;

  if (!vector_copy(H1_Z, BP_delAdelZ_H1)) return 0;
  sigmoid_prime_arr(BP_delAdelZ_H1);

  if (!vector_vector_elementwise_mult(BP_delAdelZ_H1, BP_delCdelA_H1, H1_B_grad)) return 0;
  if (!vector_vector_mult(H1_B_grad, H0, H1_W_grad)) return 0;

  if (!matrix_transpose(H1_W, BP_W_T_H1)) return 0;
  if (!matrix_vector_mult(BP_W_T_H1, H1_B_grad, H0_A_grad)) return 0;

  // HIDDEN LAYER 0
  if (!vector_copy(H0_A_grad, BP_delCdelA_H0)) return 0;

  if (!vector_copy(H0_Z, BP_delAdelZ_H0)) return 0;
  sigmoid_prime_arr(BP_delAdelZ_H0);

  if (!vector_vector_elementwise_mult(BP_delAdelZ_H0, BP_delCdelA_H0, H0_B_grad)) return 0;
  // HOTSPOT: H0_W_grad = delta_0 (x) IN   [16] outer [784] -> [16x784]
  // R: 25,088 (delta_0: 12,544 re-read per col + IN: 12,544 re-read per row)   W: 12,544   FLOPs: 12,544 (mul)
  if (!kernel_vector_vector_mult(H0_B_grad, IN, H0_W_grad)) return 0;

  return 1;
}



void feedforward() {

  // IN -> H0
  // HOTSPOT: z_0 = H0_W * IN   [16x784] * [784] -> [16]
  // R: 25,088 (H0_W: 12,544 + IN: 12,544 re-read per row)   W: 16   FLOPs: 25,088 (12,544 mul + 12,544 add)
  kernel_matrix_vector_mult(H0_W, IN, H0_mvm_res);
  vector_vector_add(H0_B, H0_mvm_res, H0_z_temp);
  vector_copy(H0_z_temp, H0_Z);
  sigmoid_arr(H0_z_temp);
  vector_copy(H0_z_temp, H0);

  // H0 -> H1
  matrix_vector_mult(H1_W, H0, H1_mvm_res);
  vector_vector_add(H1_B, H1_mvm_res, H1_z_temp);
  vector_copy(H1_z_temp, H1_Z);
  sigmoid_arr(H1_z_temp);
  vector_copy(H1_z_temp, H1);

  // H1 -> OUT
  matrix_vector_mult(L_W, H1, L_mvm_res);
  vector_vector_add(L_B, L_mvm_res, L_z_temp);
  vector_copy(L_z_temp, L_Z);
  sigmoid_arr(L_z_temp);
  vector_copy(L_z_temp, OUT);
}

void sigmoid_arr(array_ptr v) {
  for (int i = 0; i < v->len; i++) {
    v->data[i] = sigmoid(v->data[i]);
  }
}

void sigmoid_prime_arr(array_ptr v) {
  for (int i = 0; i < v->len; i++) {
    v->data[i] = sigmoid_prime(v->data[i]);
  }
}

data_t sigmoid(data_t z) {
  return 1 / (1 + pow(EULER_NUMBER_F, -z));
}

data_t sigmoid_prime(data_t z) {
  return sigmoid(z) * (1 - sigmoid(z));
}


array_ptr numToVec(int num) {
  array_ptr result = new_array(L_SIZE);  // calloc zeroes the rest
  result->data[num] = 1;
  return result;
}
