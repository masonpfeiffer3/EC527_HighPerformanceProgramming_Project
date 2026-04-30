#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "array_matrix_funcs.h"


// =====================================================================
// MATRIX
// =====================================================================

matrix_ptr new_matrix(long int rows, long int cols)
{
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;
  result->rows = rows;
  result->cols = cols;

  if (rows > 0 && cols > 0) {
    data_t *data = (data_t *) calloc(rows*cols, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                  rows * cols * sizeof(data_t));
      exit(-1);
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

long int get_matrix_rows(matrix_ptr m) { return m->rows; }
long int get_matrix_cols(matrix_ptr m) { return m->cols; }

int init_matrix(matrix_ptr m)
{
  if (m->rows > 0 && m->cols > 0) {
    for (long int i = 0; i < m->rows*m->cols; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  return 0;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)(RAND_MAX);
  return fMin + f * (fMax - fMin);
}

int init_matrix_rand(matrix_ptr m, double low, double high)
{
  if (m->rows > 0 && m->cols > 0) {
    for (long int i = 0; i < m->rows*m->cols; i++)
      m->data[i] = (data_t)(fRand(low, high));
    return 1;
  }
  return 0;
}

int zero_matrix(matrix_ptr m)
{
  if (m->rows > 0 && m->cols > 0) {
    for (long int i = 0; i < m->rows*m->cols; i++)
      m->data[i] = 0;
    return 1;
  }
  return 0;
}

data_t *get_matrix_start(matrix_ptr m) { return m->data; }


// =====================================================================
// ARRAY
// =====================================================================

array_ptr new_array(long int len)
{
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;
  result->len = len;

  if (len > 0) {
    data_t *data = (data_t *) calloc(len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      return NULL;
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

int get_array_element(array_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) return 0;
  *dest = v->data[index];
  return 1;
}

long int get_array_length(array_ptr v) { return v->len; }

int init_array(array_ptr v)
{
  if (v->len > 0) {
    for (long int i = 0; i < v->len; i++)
      v->data[i] = (data_t)(i+1);
    return 1;
  }
  return 0;
}

int init_array_rand(array_ptr v, double low, double high)
{
  if (v->len > 0) {
    for (long int i = 0; i < v->len; i++)
      v->data[i] = (data_t)(fRand(low, high));
    return 1;
  }
  return 0;
}

data_t *get_array_start(array_ptr v) { return v->data; }

int zero_array(array_ptr m)
{
  if (m->len > 0) {
    for (long int i = 0; i < m->len; i++)
      m->data[i] = 0;
    return 1;
  }
  return 0;
}


// =====================================================================
// DATASET
// =====================================================================

dataset_ptr new_dataset(long int len, long int image_len)
{
  dataset_ptr result = (dataset_ptr) malloc(sizeof(dataset));
  if (!result) return NULL;
  result->len = len;
  result->image_len = image_len;

  if (len > 0) {
    int* nums = (int*) calloc(len, sizeof(int));
    if (!nums) { free((void *) result); return NULL; }
    result->nums = nums;

    data_t* data = (data_t *) calloc(len*image_len, sizeof(data_t));
    if (!data) { free((void *) result); return NULL; }
    result->image_arr = data;
  }
  else result->image_arr = NULL;

  return result;
}

void init_dataset_rand(dataset_ptr d, double low, double high)
{
  for (long int i = 0; i < d->len; i++)
    d->nums[i] = (int)fRand(0, 10);

  for (long int i = 0; i < d->len; i++)
    for (long int j = 0; j < d->image_len; j++)
      d->image_arr[i*d->image_len + j] = (data_t)fRand(low, high);
}

void copyImageToInput(dataset_ptr d, array_ptr v, long int index)
{
  long int image_len = d->image_len;
  for (long int i = 0; i < image_len; i++)
    v->data[i] = d->image_arr[i + index*image_len];
}

// =====================================================================
// BATCH LOADER
// Copies `count` images (by indices[0..count-1]) as contiguous rows
// into `batch`.  Layout: batch[s * IMAGE_SIZE + j] = pixel j of sample s.
// =====================================================================

void copyImagesToInputBatch(dataset_ptr d, matrix_ptr batch, int *indices, int count)
{
  long int image_len = d->image_len;
  data_t  *dst       = get_matrix_start(batch);

  for (int s = 0; s < count; s++) {
    data_t *src = &d->image_arr[(long int)indices[s] * image_len];
    memcpy(dst + (long int)s * image_len, src, image_len * sizeof(data_t));
  }
}
