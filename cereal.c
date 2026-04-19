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

#define TRAIN_SIZE 50000
#define TEST_SIZE 10000

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

  // H1 ARRAY
  array_ptr H1 = new_array(H1_SIZE);
  matrix_ptr H1_W = new_matrix(H1_SIZE, H0_SIZE);
  array_ptr H1_B = new_array(H1_SIZE);

  init_array_rand(H1, 0, 1);
  init_matrix_rand(H1_W, 0, 1);
  init_array_rand(H1_B, 0, 1);

  // OUTPUT ARRAY
  array_ptr output = new_array(L_SIZE);
  matrix_ptr L_W = new_matrix(L_SIZE, H1_SIZE);
  array_ptr L_B = new_array(L_SIZE);

  init_array_rand(output, 0, 1);
  init_matrix_rand(L_W, 0, 1);
  init_array_rand(L_B, 0, 1);


  // TRAINING DATA
  dataset_ptr train_data = new_dataset(TRAIN_SIZE, I_SIZE);

  // load data into data structure
  init_dataset_rand(train_data, 0, 1);

  // STAGE 2: TRAINING



  // Load input image
  init_array_rand(input, 0, 1); 

  data_t* in = get_array_start(input);

  for (int i = 0; i < I_SIZE; i++) {
    printf("%.5f\n", in[i]);
  }



  return 0;
}






// MVM CODE (TO BE OPTIMIZED)

void matrix_vector_mult() {

}