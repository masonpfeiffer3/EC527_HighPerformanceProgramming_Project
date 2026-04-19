#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cereal.h"
#include "array_matrix_funcs.h"

#define I_SIZE 784
#define H0_SIZE 16
#define H1_SIZE 16
#define L_SIZE 10

#define TRAIN_SIZE 50
#define BATCH_SIZE 10
#define TEST_SIZE 10000

#define LEARN_RATE 1

int main() {

  // STAGE 1: LOADING DATA AND INITIALIZING ARRAYS & MATRICES

  // INPUT ARRAY
  array_ptr input = new_array(I_SIZE);
  
  // H0 ARRAY, WEIGHTS, and BIASES
  array_ptr H0 = new_array(H0_SIZE);
  matrix_ptr H0_W = new_matrix(H0_SIZE, I_SIZE);
  array_ptr H0_B = new_array(H0_SIZE);

  init_array_rand(H0, 0, 1);
  init_matrix_rand(H0_W, 0, 1);
  init_array_rand(H0_B, 0, 1);

  // H1 ARRAY, WEIGHTS, and BIASES
  array_ptr H1 = new_array(H1_SIZE);
  matrix_ptr H1_W = new_matrix(H1_SIZE, H0_SIZE);
  array_ptr H1_B = new_array(H1_SIZE);

  init_array_rand(H1, 0, 1);
  init_matrix_rand(H1_W, 0, 1);
  init_array_rand(H1_B, 0, 1);

  // OUTPUT ARRAY, WEIGHTS, and BIASES
  array_ptr output = new_array(L_SIZE);
  matrix_ptr L_W = new_matrix(L_SIZE, H1_SIZE);
  array_ptr L_B = new_array(L_SIZE);

  init_array_rand(output, 0, 1);
  init_matrix_rand(L_W, 0, 1);
  init_array_rand(L_B, 0, 1);


  // TRAINING DATA
  dataset_ptr train_data = new_dataset(TRAIN_SIZE, I_SIZE);

  // load data into data structure
  init_dataset_rand(train_data, 0, 1);  // TO BE REPLACED


  // STAGE 2: TRAINING

  for (int i = 0; i < TRAIN_SIZE; i++) {

    // load input from dataset
    copyImageToInput(train_data, input, i);

    int num = train_data->nums[i];









  }



  return 0;
}


data_t sigmoid(data_t z) {
  return 1.0 / (1.0 + exp(-z));
}

data_t sigmoid_prime(data_t z) {
  return sigmoid(z) * (1 - sigmoid(z));
}



// MVM CODE (TO BE OPTIMIZED)
int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out) {

  long int rows = get_matrix_rows(m);
  long int cols = get_matrix_cols(m);
  long int vlen = get_array_length(v);

  data_t* weights = get_matrix_start(m);
  data_t* lastLayerActivations = get_array_start(v);
  data_t* v_out_loc = get_array_start(v_out);

  data_t sum = 0;

  if (vlen == cols) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        sum += weights[i*cols + j] * lastLayerActivations[j];
      }
      v_out_loc[i] = sum;
      sum = 0;
    }
    return 1;
  }

  return 0;

}

int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutlen = get_array_length(v_out);

  data_t* v1_start = get_array_start(v1);
  data_t* v2_start = get_array_start(v2);
  data_t* vout_start = get_array_start(v_out);

  if (v1len == v2len && v2len == voutlen) {
    for (int i = 0; i < v1len; i++) {
      v_out[i] = v1_start[i] + v2_start[i];
    }
    return 1;
  }

  return 0;

}