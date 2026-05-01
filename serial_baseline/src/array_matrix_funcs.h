#ifndef ARRAY_MATRIX_FUNCS_H
#define ARRAY_MATRIX_FUNCS_H

#include "params.h"

// MATRIX
// creation, initialization, and access
matrix_ptr new_matrix(long int rows, long int cols); //creates a flattened 2D matrix, initialized to 0s
long int get_matrix_rows(matrix_ptr m);
long int get_matrix_cols(matrix_ptr m);
int init_matrix(matrix_ptr m); //initialize matrix to 0, 1, 2
double fRand(double fMin, double fMax); //generate random double between fMin and fMax
int init_matrix_rand(matrix_ptr m, double low, double high); //initialize matrix to random
int zero_matrix(matrix_ptr m); //set all elements of matrix to 0
data_t *get_matrix_start(matrix_ptr m); //return pointer to start of matrix data



// array func declarations
array_ptr new_array(long int len); //created 1D array, initialized to 0s
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
data_t *get_array_start(array_ptr v);
int init_array(array_ptr v);
int zero_array(array_ptr m);
int init_array_rand(array_ptr v, double low, double high);


// dataset func declarations
dataset_ptr new_dataset(long int len, long int image_len);
void init_dataset_rand(dataset_ptr d, double low, double high);
void copyImageToInput(dataset_ptr d, array_ptr v, long int index);

#endif