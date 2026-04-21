#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "array_matrix_math.h"
#include "array_matrix_funcs.h"


int matrix_transpose(matrix_ptr m, matrix_ptr m_out) {

  long int rows = get_matrix_rows(m);
  long int cols = get_matrix_cols(m);
  long int out_rows = get_matrix_rows(m_out);
  long int out_cols = get_matrix_cols(m_out);

  data_t* m_start = get_matrix_start(m);
  data_t* m_out_start = get_matrix_start(m_out);

  if (rows == out_cols && cols == out_rows) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        m_out_start[j*out_cols + i] = m_start[i*cols + j];
      }
    }
    return 1;
  }

  return 0;

}

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
//matrix matrix add
int matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out){
  
  long int rows1 = get_matrix_rows(m1);
  long int cols1 = get_matrix_cols(m1);
  long int rows2 = get_matrix_rows(m2);
  long int cols2 = get_matrix_cols(m2);
  long int rows_out = get_matrix_rows(m_out);
  long int cols_out = get_matrix_cols(m_out);

  data_t* m1_start = get_matrix_start(m1);
  data_t* m2_start = get_matrix_start(m2);
  data_t* m_out_start = get_matrix_start(m_out);

  if (rows1 == rows2 && cols1 == cols2 && rows1 == rows_out && cols1 == cols_out) {
    for (int i = 0; i < rows1; i++) {
      for (int j = 0; j < cols1; j++) {
        m_out_start[i*cols_out + j] = m1_start[i*cols1 + j] + m2_start[i*cols2 + j];
      }
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

//vector vector sub
int vector_vector_sub(array_ptr v1, array_ptr v2, array_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutlen = get_array_length(v_out);

  data_t* v1_start = get_array_start(v1);
  data_t* v2_start = get_array_start(v2);
  data_t* vout_start = get_array_start(v_out);

  if (v1len == v2len && v2len == voutlen) {
    for (int i = 0; i < v1len; i++) {
      vout_start[i] = v1_start[i] - v2_start[i];
    }
    return 1;
  }

  return 0;

}
//standard vector multiply
int vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutrow = get_matrix_rows(v_out);
  int voutcol = get_matrix_cols(v_out);
  

  data_t* v1_start = get_array_start(v1);
  data_t* v2_start = get_array_start(v2);
  data_t* vout_start = get_matrix_start(v_out);

  if (v1len == voutrow && v2len == voutcol) {
    for (int i = 0; i < v1len; i++) {
      for(int j = 0; j < v2len; j++) {
        vout_start[i*voutcol + j] = v1_start[i] * v2_start[j];
      }
    }
    return 1;
  }

  return 0;

}

//vector vector elementwise mult
int vector_vector_elementwise_mult(array_ptr v1, array_ptr v2, array_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutlen = get_array_length(v_out);

  data_t* v1_start = get_array_start(v1);
  data_t* v2_start = get_array_start(v2);
  data_t* vout_start = get_array_start(v_out);

  if (v1len == v2len && v2len == voutlen) {
    for (int i = 0; i < v1len; i++) {
      vout_start[i] = v1_start[i] * v2_start[i];
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

