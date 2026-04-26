#ifndef CEREAL_H
#define CEREAL_H
#include "array_matrix_funcs.h"

void serial_MNIST(dataset_ptr train_data, dataset_ptr test_data);
void train_MNIST(dataset_ptr train_data);
void test_MNIST(dataset_ptr test_data);
void init_MNIST(void);

int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);
int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_copy(array_ptr source, array_ptr dest);
data_t sigmoid(data_t z);
void sigmoid_arr(array_ptr v);
data_t sigmoid_prime(data_t z);
void sigmoid_prime_arr(array_ptr v);
void feedforward();
int backprop(int num);
array_ptr numToVec(int num);
#endif