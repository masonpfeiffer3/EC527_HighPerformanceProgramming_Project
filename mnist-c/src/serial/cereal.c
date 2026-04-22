#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../params.h"
#include "array_matrix_funcs.h"
#include "array_matrix_math.h"
#include "cereal.h"

#define I_SIZE  IMAGE_SIZE
#define H0_SIZE 16
#define H1_SIZE 16
#define L_SIZE  10

#define BATCH_SIZE 10
#define LEARN_RATE 1

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

  IN = new_array(I_SIZE);

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
}

void train_MNIST(dataset_ptr train_data){

  // Per-batch gradient accumulators
  matrix_ptr H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  matrix_ptr H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  matrix_ptr L_W_grad_sum  = new_matrix(L_SIZE, H1_SIZE);

  array_ptr H0_B_grad_sum = new_array(H0_SIZE);
  array_ptr H1_B_grad_sum = new_array(H1_SIZE);
  array_ptr L_B_grad_sum  = new_array(L_SIZE);

  for (int i = 0; i < TRAIN_SIZE; i += BATCH_SIZE) {

    // printf("BATCH #%d starting\n", i / BATCH_SIZE);

    // Accumulate gradients across the batch
    for (int j = i; j < i + BATCH_SIZE; j++) {
      copyImageToInput(train_data, IN, j);
      int num = train_data->nums[j];

      // printf("Num: %d\n", num);

      feedforward();

      // DEBUG: output layer activations after feedforward
      printf("last layer activations: ");
      for (int k = 0; k < OUT->len; k++) printf("%f, ", OUT->data[k]);
      printf("\n");

      if (!backprop(num)) printf("backprop failed!");

      //--- DEBUG: inspect output layer state after backprop ---
      // printf("last layer biases: ");
      // for (int k = 0; k < L_B->len; k++) printf("%f, ", L_B->data[k]);
      // printf("\n");
      
      // printf("last layer L_B_grad_sum: ");
      // for (int k = 0; k < L_B_grad_sum->len; k++) printf("%f, ", L_B_grad_sum->data[k]);
      // printf("\n");
      
      // printf("2nd element of L_W_grad: %f\n", L_W_grad->data[1]);

      matrix_matrix_add(H0_W_grad_sum, H0_W_grad, H0_W_grad_sum);
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

void test_MNIST(dataset_ptr test_data) {
  int correct = 0;
  int wrong = 0;

  for (int i = 0; i < TEST_SIZE; i++) {
    // plug in each of the tests into the parameter set
    copyImageToInput(test_data, IN, i);
    feedforward();

    // check the output, use the max function
    output_max prediction = vector_max(OUT);
    int label = test_data->nums[i];

    // if the index matches the label, accumulate a correct; if not accumulate a wrong
    if (prediction.index == label) {
      correct++;
    } else {
      wrong++;
    }
  }

  // determine accuracy
  data_t accuracy = (data_t)correct / TEST_SIZE;
  printf("Test accuracy: %d / %d (%.2f%%)\n", correct, TEST_SIZE, accuracy * 100);
}


int backprop(int num) {

  // OUTPUT LAYER 
  // dC/dA = OUT - y
  array_ptr delCdelA = new_array(L_SIZE);
  array_ptr y = numToVec(num);
  if (!vector_vector_sub(OUT, y, delCdelA)) return 0;

  // DEBUG: output layer gradient + target 
  // printf("L delCdelA 5: %f\n", delCdelA->data[4]);
  // printf("L delCdelA 4: %f\n", delCdelA->data[3]);
  // printf("y 5: %f\n", y->data[5]);

  // dA/dZ = sigmoid'(Z)
  array_ptr delAdelZ = new_array(L_SIZE);
  if (!vector_copy(L_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);

  // Bias gradient: dC/dB = dA/dZ * dC/dA
  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, L_B_grad)) return 0;

  // Weight gradient: outer product of bias grad with previous activations
  if (!vector_vector_mult(L_B_grad, H1, L_W_grad)) return 0;

  // Propagate to H1 activation gradient
  matrix_ptr W_transpose = new_matrix(H1_SIZE, L_SIZE);
  if (!matrix_transpose(L_W, W_transpose)) return 0;
  if (!matrix_vector_mult(W_transpose, L_B_grad, H1_A_grad)) return 0;


  // HIDDEN LAYER 1 
  delCdelA = new_array(H1_SIZE);
  if (!vector_copy(H1_A_grad, delCdelA)) return 0;

  // DEBUG: H1 layer gradient
  // printf("H1 delCdelA 5: %f\n", delCdelA->data[4]);
  // printf("H1 delCdelA 4: %f\n", delCdelA->data[3]);

  delAdelZ = new_array(H1_SIZE);
  if (!vector_copy(H1_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);

  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, H1_B_grad)) return 0;
  if (!vector_vector_mult(H1_B_grad, H0, H1_W_grad)) return 0;

  W_transpose = new_matrix(H0_SIZE, H1_SIZE);
  if (!matrix_transpose(H1_W, W_transpose)) return 0;
  if (!matrix_vector_mult(W_transpose, H1_B_grad, H0_A_grad)) return 0;


  // HIDDEN LAYER 0
  delCdelA = new_array(H0_SIZE);
  if (!vector_copy(H0_A_grad, delCdelA)) return 0;

  // DEBUG: H0 layer gradient
  // printf("H0 delCdelA 5: %f\n", delCdelA->data[4]);
  // printf("H0 delCdelA 4: %f\n", delCdelA->data[3]);

  delAdelZ = new_array(H0_SIZE);
  if (!vector_copy(H0_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);

  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, H0_B_grad)) return 0;
  if (!vector_vector_mult(H0_B_grad, IN, H0_W_grad)) return 0;

  return 1;
}



int feedforward() {
  // DEBUG: shape check before first matmul
  // printf("H0_W dim: %d x %d --- input len: %d\n", H0_W->rows, H0_W->cols, IN->len);

  // IN -> H0
  array_ptr mvm_res = new_array(H0_SIZE);
  array_ptr z       = new_array(H0_SIZE);
  if (!matrix_vector_mult(H0_W, IN, mvm_res)) return 0;
  if (!vector_vector_add(H0_B, mvm_res, z))  return 0;
  vector_copy(z, H0_Z);
  sigmoid_arr(z);
  if (!vector_copy(z, H0)) return 0;

  // H0 -> H1
  mvm_res = new_array(H1_SIZE);
  z       = new_array(H1_SIZE);
  matrix_vector_mult(H1_W, H0, mvm_res);
  vector_vector_add(H1_B, mvm_res, z);
  vector_copy(z, H1_Z);
  sigmoid_arr(z);
  vector_copy(z, H1);

  // H1 -> OUT
  mvm_res = new_array(L_SIZE);
  z       = new_array(L_SIZE);
  matrix_vector_mult(L_W, H1, mvm_res);
  vector_vector_add(L_B, mvm_res, z);
  vector_copy(z, L_Z);
  sigmoid_arr(z);
  vector_copy(z, OUT);
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