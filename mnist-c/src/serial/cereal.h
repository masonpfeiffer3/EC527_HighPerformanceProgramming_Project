#ifndef CEREAL_H
#define CEREAL_H

#include "array_matrix_funcs.h"


int serial_MNIST();


int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);
int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_copy(array_ptr source, array_ptr dest);

data_t sigmoid(data_t z);
void sigmoid_arr(array_ptr v);
data_t sigmoid_prime(data_t z);

int feedforward();
array_ptr numToVec(int num);

#endif