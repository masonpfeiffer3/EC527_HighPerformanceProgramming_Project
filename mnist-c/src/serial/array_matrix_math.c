#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array_matrix_math.h"
#include ".../params.h"

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
      vout_start[i] = v1_start[i] + v2_start[i];
    }
    return 1;
  }

  return 0;

}

int vector_copy(array_ptr source, array_ptr dest) {
  int v1len = get_array_length(source);
  int v2len = get_array_length(dest);

  data_t* v1_start = get_array_start(source);
  data_t* v2_start = get_array_start(dest);

  if (v1len == v2len) {
    for (int i = 0; i < v1len; i++) {
      v2_start[i] = v1_start[i];
    }
    return 1;
  }

  return 0;
}

array_ptr numToVec(int num) {
  array_ptr result = new_array(L_SIZE);  // initializes all elements to 0 with calloc

  result->data[num] = 1;

  return result;
}