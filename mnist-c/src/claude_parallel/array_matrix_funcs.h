#ifndef ARRAY_MATRIX_FUNCS_H
#define ARRAY_MATRIX_FUNCS_H

#include "../params.h"

// MATRIX
matrix_ptr new_matrix(long int rows, long int cols);
long int get_matrix_rows(matrix_ptr m);
long int get_matrix_cols(matrix_ptr m);
int init_matrix(matrix_ptr m);
double fRand(double fMin, double fMax);
int init_matrix_rand(matrix_ptr m, double low, double high);
int zero_matrix(matrix_ptr m);
data_t *get_matrix_start(matrix_ptr m);

// ARRAY
array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
data_t *get_array_start(array_ptr v);
int init_array(array_ptr v);
int zero_array(array_ptr m);
int init_array_rand(array_ptr v, double low, double high);

// DATASET
dataset_ptr new_dataset(long int len, long int image_len);
void init_dataset_rand(dataset_ptr d, double low, double high);
void copyImageToInput(dataset_ptr d, array_ptr v, long int index);

// BATCH LOADER
// Copies `count` images (selected by indices[0..count-1]) into the first
// `count` rows of `batch` (layout: count x IMAGE_SIZE, samples as rows).
void copyImagesToInputBatch(dataset_ptr d, matrix_ptr batch, int *indices, int count);

#endif
