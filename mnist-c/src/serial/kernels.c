#include "kernels.h"
#include "array_matrix_funcs.h"

int kernel_matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out) {

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

int kernel_vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out) {

  int v1len = get_array_length(v1);
  int v2len = get_array_length(v2);
  int voutrow = get_matrix_rows(v_out);
  int voutcol = get_matrix_cols(v_out);

  data_t* v1_start = get_array_start(v1);
  data_t* v2_start = get_array_start(v2);
  data_t* vout_start = get_matrix_start(v_out);

  data_t v1_val;

  if (v1len == voutrow && v2len == voutcol) {
    for (int i = 0; i < v1len; i++) {
      v1_val = v1_start[i];
      for (int j = 0; j < v2len; j+=16) {
        vout_start[i*voutcol + j] = v1_val * v2_start[j];
        vout_start[i*voutcol + j + 1] = v1_val * v2_start[j + 1];
        vout_start[i*voutcol + j + 2] = v1_val * v2_start[j + 2];
        vout_start[i*voutcol + j + 3] = v1_val * v2_start[j + 3];
        vout_start[i*voutcol + j + 4] = v1_val * v2_start[j + 4];
        vout_start[i*voutcol + j + 5] = v1_val * v2_start[j + 5];
        vout_start[i*voutcol + j + 6] = v1_val * v2_start[j + 6];
        vout_start[i*voutcol + j + 7] = v1_val * v2_start[j + 7];
        vout_start[i*voutcol + j + 8] = v1_val * v2_start[j + 8];
        vout_start[i*voutcol + j + 9] = v1_val * v2_start[j + 9];
        vout_start[i*voutcol + j + 10] = v1_val * v2_start[j + 10];
        vout_start[i*voutcol + j + 11] = v1_val * v2_start[j + 11];
        vout_start[i*voutcol + j + 12] = v1_val * v2_start[j + 12];
        vout_start[i*voutcol + j + 13] = v1_val * v2_start[j + 13];
        vout_start[i*voutcol + j + 14] = v1_val * v2_start[j + 14];
        vout_start[i*voutcol + j + 15] = v1_val * v2_start[j + 15];
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

  data_t* m1_start = get_matrix_start(m1);
  data_t* m2_start = get_matrix_start(m2);
  data_t* m_out_start = get_matrix_start(m_out);

  if (rows1 == rows2 && cols1 == cols2 && rows1 == rows_out && cols1 == cols_out) {
    for (int i = 0; i < rows1; i++) {
      for (int j = 0; j < cols1; j+=16) {
        m_out_start[i*cols_out + j] = m1_start[i*cols1 + j] + m2_start[i*cols2 + j];
        m_out_start[i*cols_out + j + 1] = m1_start[i*cols1 + j + 1] + m2_start[i*cols2 + j + 1];
        m_out_start[i*cols_out + j + 2] = m1_start[i*cols1 + j + 2] + m2_start[i*cols2 + j + 2];
        m_out_start[i*cols_out + j + 3] = m1_start[i*cols1 + j + 3] + m2_start[i*cols2 + j + 3];
        m_out_start[i*cols_out + j + 4] = m1_start[i*cols1 + j + 4] + m2_start[i*cols2 + j + 4];
        m_out_start[i*cols_out + j + 5] = m1_start[i*cols1 + j + 5] + m2_start[i*cols2 + j + 5];
        m_out_start[i*cols_out + j + 6] = m1_start[i*cols1 + j + 6] + m2_start[i*cols2 + j + 6];
        m_out_start[i*cols_out + j + 7] = m1_start[i*cols1 + j + 7] + m2_start[i*cols2 + j + 7];
        m_out_start[i*cols_out + j + 8] = m1_start[i*cols1 + j + 8] + m2_start[i*cols2 + j + 8];
        m_out_start[i*cols_out + j + 9] = m1_start[i*cols1 + j + 9] + m2_start[i*cols2 + j + 9];
        m_out_start[i*cols_out + j + 10] = m1_start[i*cols1 + j + 10] + m2_start[i*cols2 + j + 10];
        m_out_start[i*cols_out + j + 11] = m1_start[i*cols1 + j + 11] + m2_start[i*cols2 + j + 11];
        m_out_start[i*cols_out + j + 12] = m1_start[i*cols1 + j + 12] + m2_start[i*cols2 + j + 12];
        m_out_start[i*cols_out + j + 13] = m1_start[i*cols1 + j + 13] + m2_start[i*cols2 + j + 13];
        m_out_start[i*cols_out + j + 14] = m1_start[i*cols1 + j + 14] + m2_start[i*cols2 + j + 14];
        m_out_start[i*cols_out + j + 15] = m1_start[i*cols1 + j + 15] + m2_start[i*cols2 + j + 15];
      }
    }
    return 1;
  }

  return 0;
}
