#include "kernels.h"
#include "array_matrix_funcs.h"

int kernel_matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out) {

  long int rows = get_matrix_rows(m);
  long int cols = get_matrix_cols(m);
  long int vlen = get_array_length(v);

  data_t* restrict weights = get_matrix_start(m);
  data_t* restrict lastLayerActivations = get_array_start(v);
  data_t* restrict v_out_loc = get_array_start(v_out);

  data_t sum0 = 0;
  data_t sum1 = 0;
  data_t sum2 = 0;
  data_t sum3 = 0;
  data_t sum4 = 0;
  data_t sum5 = 0;

  int i, j, row_offset, offset;

  if (vlen == cols) {
    for (i = 0; i < rows; i++) {
      
      row_offset = i*cols;

      for (j = 0; j < cols - 5; j+=6) {
        offset = row_offset + j;
        sum0 += weights[offset] * lastLayerActivations[j];
        sum1 += weights[offset + 1] * lastLayerActivations[j + 1];
        sum2 += weights[offset + 2] * lastLayerActivations[j + 2];
        sum3 += weights[offset + 3] * lastLayerActivations[j + 3];
        sum4 += weights[offset + 4] * lastLayerActivations[j + 4];
        sum5 += weights[offset + 5] * lastLayerActivations[j + 5];
      }

      // finish remaining elements
      for (; j < cols; j++) {
        sum0 += weights[row_offset + j] * lastLayerActivations[j];
      }

      v_out_loc[i] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5;
      sum0 = 0;
      sum1 = 0;
      sum2 = 0;
      sum3 = 0;
      sum4 = 0;
      sum5 = 0;
    }
    return 1;
  }

  return 0;
}

int kernel_vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutrow = get_matrix_rows(v_out);
  int voutcol = get_matrix_cols(v_out);

  data_t* restrict v1_start = get_array_start(v1);
  data_t* restrict v2_start = get_array_start(v2);
  data_t* restrict vout_start = get_matrix_start(v_out);

  data_t v1_val;

  int i, j, row_offset, offset;

  if (v1len == voutrow && v2len == voutcol) {
    for (i = 0; i < v1len; i++) {
      v1_val = v1_start[i];

      row_offset = i*voutcol;

      for (j = 0; j < v2len - 5; j+=6) {
        offset = row_offset + j;
        vout_start[offset] = v1_val * v2_start[j];
        vout_start[offset + 1] = v1_val * v2_start[j + 1];
        vout_start[offset + 2] = v1_val * v2_start[j + 2];
        vout_start[offset + 3] = v1_val * v2_start[j + 3];
        vout_start[offset + 4] = v1_val * v2_start[j + 4];
        vout_start[offset + 5] = v1_val * v2_start[j + 5];
      }
      
      // finish up remaining elements
      for (; j < v2len; j++) {
        vout_start[row_offset + j] = v1_val * v2_start[j];
      }


    }
    return 1;
  }

  return 0;
}

int kernel_matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out) {

  long int rows1 = get_matrix_rows(m1);
  long int cols1 = get_matrix_cols(m1);
  long int rows2 = get_matrix_rows(m2);
  long int cols2 = get_matrix_cols(m2);
  long int rows_out = get_matrix_rows(m_out);
  long int cols_out = get_matrix_cols(m_out);

  data_t* restrict m1_start = get_matrix_start(m1);
  data_t* restrict m2_start = get_matrix_start(m2);
  data_t* restrict m_out_start = get_matrix_start(m_out);

  int i, j, row_offset, offset;

  if (rows1 == rows2 && cols1 == cols2 && rows1 == rows_out && cols1 == cols_out) {
    for (i = 0; i < rows1; i++) {

      row_offset = i*cols1;

      for (j = 0; j < cols1 - 5; j+=6) {
        offset = row_offset + j;
        m_out_start[offset] = m1_start[offset] + m2_start[offset];
        m_out_start[offset + 1] = m1_start[offset + 1] + m2_start[offset + 1];
        m_out_start[offset + 2] = m1_start[offset + 2] + m2_start[offset + 2];
        m_out_start[offset + 3] = m1_start[offset + 3] + m2_start[offset + 3];
        m_out_start[offset + 4] = m1_start[offset + 4] + m2_start[offset + 4];
        m_out_start[offset + 5] = m1_start[offset + 5] + m2_start[offset + 5];
      }

      // finish up remaining elements
      for (; j < cols1; j++) {
        m_out_start[row_offset + j] = m1_start[row_offset + j] + m2_start[row_offset + j];
      }

    }


    return 1;
  }

  return 0;
}
