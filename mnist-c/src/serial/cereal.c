#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../params.h"
#include "array_matrix_funcs.h"
#include "array_matrix_math.h"
#include "cereal.h"

#define I_SIZE IMAGE_SIZE
#define H0_SIZE 16
#define H1_SIZE 16
#define L_SIZE 10

#define BATCH_SIZE 1

#define LEARN_RATE 1

// GLOBAL LAYERS -------------------------------------------------------------------

// INPUT ARRAY
array_ptr IN;
  
// H0 ARRAY, WEIGHTS, BIASES, Z, and gradients
array_ptr H0;
matrix_ptr H0_W;
array_ptr H0_B;
array_ptr H0_Z;

matrix_ptr H0_W_grad;
array_ptr H0_B_grad;

// H1 ARRAY, WEIGHTS, BIASES, Z, and gradients
array_ptr H1;
matrix_ptr H1_W;
array_ptr H1_B;
array_ptr H1_Z;

matrix_ptr H1_W_grad;
array_ptr H1_B_grad;
array_ptr H0_A_grad;

// OUTPUT ARRAY, WEIGHTS, BIASES, Z, and gradients
array_ptr OUT;
matrix_ptr L_W;
array_ptr L_B;
array_ptr L_Z;

matrix_ptr L_W_grad;
array_ptr L_B_grad;
array_ptr H1_A_grad;

// SERIAL MNIST ---------------------------------------------------------------------

int serial_MNIST(dataset_ptr train_data, dataset_ptr test_data) {

  // STAGE 1: LOADING DATA AND INITIALIZING ARRAYS & MATRICES

  // INPUT ARRAY
  IN = new_array(I_SIZE);
    
  // H0 ARRAY, WEIGHTS, and BIASES
  H0 = new_array(H0_SIZE);
  H0_W = new_matrix(H0_SIZE, I_SIZE);
  H0_B = new_array(H0_SIZE);
  H0_Z = new_array(H0_SIZE);

  H0_W_grad = new_matrix(H0_SIZE, I_SIZE);
  H0_B_grad = new_array(H0_SIZE);

  init_array_rand(H0, 0, 1);
  init_matrix_rand(H0_W, -1, 1);
  init_array_rand(H0_B, -1, 1);

  // H1 ARRAY, WEIGHTS, and BIASES
  H1 = new_array(H1_SIZE);
  H1_W = new_matrix(H1_SIZE, H0_SIZE);
  H1_B = new_array(H1_SIZE);
  H1_Z = new_array(H1_SIZE);

  H1_W_grad = new_matrix(H1_SIZE, H0_SIZE);
  H1_B_grad = new_array(H1_SIZE);
  H0_A_grad = new_array(H0_SIZE);

  init_array_rand(H1, 0, 1);
  init_matrix_rand(H1_W, -1, 1);
  init_array_rand(H1_B, -1, 1);

  // OUTPUT ARRAY, WEIGHTS, and BIASES
  OUT = new_array(L_SIZE);
  L_W = new_matrix(L_SIZE, H1_SIZE);
  L_B = new_array(L_SIZE);
  L_Z = new_array(L_SIZE);

  L_W_grad = new_matrix(L_SIZE, H1_SIZE);
  L_B_grad = new_array(L_SIZE);
  H1_A_grad = new_array(H1_SIZE);

  init_array_rand(OUT, 0, 1);
  init_matrix_rand(L_W, -1, 1);
  init_array_rand(L_B, -1, 1);


  // // TRAINING DATA
  // dataset_ptr train_data = new_dataset(TRAIN_SIZE, I_SIZE);

  // // load data into data structure
  // init_dataset_rand(train_data, 0, 1);  // TO BE REPLACED


  // STAGE 2: TRAINING

  matrix_ptr H0_W_grad_sum = new_matrix(H0_SIZE, I_SIZE);
  matrix_ptr H1_W_grad_sum = new_matrix(H1_SIZE, H0_SIZE);
  matrix_ptr L_W_grad_sum = new_matrix(L_SIZE, H1_SIZE);

  array_ptr H0_B_grad_sum = new_array(H0_SIZE);
  array_ptr H1_B_grad_sum = new_array(H1_SIZE);
  array_ptr L_B_grad_sum = new_array(L_SIZE);

  for (int i = 0; i < TRAIN_SIZE; i+=BATCH_SIZE) {

    for (int j = i; j < i + BATCH_SIZE; j++) {

      copyImageToInput(train_data, IN, i);   // load input from dataset

      int num = train_data->nums[i];   // load number


      printf("Num: %d\n", num);

      feedforward();  // FEEDFORWARD

      //printf("last layer second element (1): %f\n", output->data[1]);

      if (!backprop(num)) printf("backprop failed!");  // BACKPROPAGATION

      printf("2nd element of L_W_grad: %f\n", L_W_grad->data[1]);

      matrix_matrix_add(H0_W_grad_sum, H0_W_grad, H0_W_grad_sum);
      matrix_matrix_add(H1_W_grad_sum, H1_W_grad, H1_W_grad_sum);
      matrix_matrix_add(L_W_grad_sum, L_W_grad, L_W_grad_sum);

      vector_vector_add(H0_B_grad_sum, H0_B_grad, H0_B_grad_sum);
      vector_vector_add(H1_B_grad_sum, H1_B_grad, H1_B_grad_sum);
      vector_vector_add(H1_B_grad_sum, H1_B_grad, H1_B_grad_sum);

    }



  }



  return 0;
}


int backprop(int num) {

  // OUTPUT LAYER
  // bias gradient
  array_ptr delCdelA = new_array(L_SIZE);
  array_ptr y = numToVec(num);
  if (!vector_vector_sub(OUT, y, delCdelA)) return 0;  // set delCdelA
  printf("L delCdelA 5: %f\n", delCdelA->data[4]);
  printf("L delCdelA 4: %f\n", delCdelA->data[3]);
  //printf("y 5: %f\n", y->data[5]);
  
  array_ptr delAdelZ = new_array(L_SIZE);
  if (!vector_copy(L_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);   // set delAdelZ

  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, L_B_grad)) return 0;

  // weight gradient
  if (!vector_vector_mult(L_B_grad, H1, L_W_grad)) return 0;  // delZdelW are the activations of H1

  // H1 layer activation gradient
  matrix_ptr W_transpose = new_matrix(H1_SIZE, L_SIZE);
  if (!matrix_transpose(L_W, W_transpose)) return 0;
  if(!matrix_vector_mult(W_transpose, L_B_grad, H1_A_grad)) return 0;


  // H1 LAYER
  // bias gradient
  delCdelA = new_array(H1_SIZE);
  if (!vector_copy(H1_A_grad, delCdelA)) return 0;  // set delCdelA
  printf("H1 delCdelA 5: %f\n", delCdelA->data[4]);
  printf("H1 delCdelA 4: %f\n", delCdelA->data[3]);
  
  delAdelZ = new_array(H1_SIZE);
  if (!vector_copy(H1_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);   // set delAdelZ

  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, H1_B_grad)) return 0;

  // weight gradient
  if (!vector_vector_mult(H1_B_grad, H0, H1_W_grad)) return 0;  // delZdelW are the activations of H0

  // H1 layer activation gradient
  W_transpose = new_matrix(H0_SIZE, H1_SIZE);
  if (!matrix_transpose(H1_W, W_transpose)) return 0;
  if (!matrix_vector_mult(W_transpose, H1_B_grad, H0_A_grad)) return 0;


  // H0 LAYER
  //bias gradient
  delCdelA = new_array(H0_SIZE);
  if (!vector_copy(H0_A_grad, delCdelA)) return 0;  // set delCdelA
  printf("H0 delCdelA 5: %f\n", delCdelA->data[4]);
  printf("H0 delCdelA 4: %f\n", delCdelA->data[3]);
  
  delAdelZ = new_array(H0_SIZE);
  if (!vector_copy(H0_Z, delAdelZ)) return 0;
  sigmoid_prime_arr(delAdelZ);   // set delAdelZ

  if (!vector_vector_elementwise_mult(delAdelZ, delCdelA, H0_B_grad)) return 0;

  // weight gradient
  if (!vector_vector_mult(H0_B_grad, IN, H0_W_grad)) return 0;  // delZdelW are the activations of IN

  return 1;

}



int feedforward() {
  // LAYER 0 -> LAYER 1
  array_ptr mvm_res = new_array(H0_SIZE);
  array_ptr z = new_array(H0_SIZE);
  //printf("H0_W dim: %d x %d --- input len: %d\n", H0_W->rows, H0_W->cols, input->len);
  if (!matrix_vector_mult(H0_W, IN, mvm_res)) return 0;  // troubleshooting purposes
  if (!vector_vector_add(H0_B, mvm_res, z)) return 0;
  vector_copy(z, H0_Z);
  sigmoid_arr(z);
  if (!vector_copy(z, H0)) return 0;
  
  // LAYER 1 -> LAYER 2
  mvm_res = new_array(H1_SIZE);
  z = new_array(H1_SIZE);
  matrix_vector_mult(H1_W, H0, mvm_res);
  vector_vector_add(H1_B, mvm_res, z);
  vector_copy(z, H1_Z);
  sigmoid_arr(z);
  vector_copy(z, H1);

  // LAYER 2 -> LAYER 3
  mvm_res = new_array(L_SIZE);
  z = new_array(L_SIZE);
  matrix_vector_mult(L_W, H1, mvm_res);
  vector_vector_add(L_B, mvm_res, z);
  vector_copy(z, L_Z);
  sigmoid_arr(z);
  vector_copy(z, OUT);

}


void sigmoid_arr(array_ptr v) {
  int len = v->len;

  for (int i = 0; i < len; i++) {
    v->data[i] = sigmoid(v->data[i]);
  }
}

void sigmoid_prime_arr(array_ptr v) {
  int len = v->len;

  for (int i = 0; i < len; i++) {
    v->data[i] = sigmoid_prime(v->data[i]);
  }
}


data_t sigmoid(data_t z) {
    return (1 / (1 + pow(EULER_NUMBER_F, -z)));
}

data_t sigmoid_prime(data_t z) {
  return sigmoid(z) * (1 - sigmoid(z));
}



array_ptr numToVec(int num) {
  array_ptr result = new_array(L_SIZE);  // initializes all elements to 0 with calloc

  result->data[num] = 1;

  return result;
  //vector_copy(result, y);
}



