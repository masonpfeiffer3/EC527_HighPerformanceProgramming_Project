#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>           /* AVX, AVX2 */

#include "array_matrix_funcs.h"

#define AVX_STRIDE 8


// MATRIX STRUCT FUNCTIONS ----- BORROWED FROM PROF. HERBORDT


matrix_ptr new_matrix(long int rows, long int cols)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->rows = rows;
  result->cols = cols;

  /* Allocate and declare array */
  if (rows > 0 && cols > 0) {
    data_t *data = (data_t *) calloc(rows*cols, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                  rows * cols * sizeof(data_t) );
      exit(-1);
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Return row length of matrix */
long int get_matrix_rows(matrix_ptr m)
{
  return m->rows;
}

/* Return row length of matrix */
long int get_matrix_cols(matrix_ptr m)
{
  return m->cols;
}


/* initialize matrix */
int init_matrix(matrix_ptr m)
{
  long int i;

  if (m->rows > 0 && m->cols > 0) {
    for (i = 0; i < m->rows*m->cols; i++) {
      m->data[i] = (data_t)(i);
    }
    return 1;
  }
  else return 0;
}

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)(RAND_MAX);
  return fMin + f * (fMax - fMin);
}

/* initialize matrix to rand */
int init_matrix_rand(matrix_ptr m, double low, double high)
{
  long int i;

  if (m->rows > 0 && m->cols > 0) {
    for (i = 0; i < m->rows*m->cols; i++) {
      m->data[i] = (data_t)(fRand(low, high));
    }
    return 1;
  }
  else return 0;
}

int zero_matrix(matrix_ptr m)
{
  if (m->rows > 0 && m->cols > 0) {
    long int total = m->rows * m->cols;
    data_t* restrict d = m->data;
    __m256 zv = _mm256_setzero_ps();
    long int i;
    for (i = 0; i <= total - AVX_STRIDE; i += AVX_STRIDE) {
      _mm256_storeu_ps(&d[i], zv);
    }
    for (; i < total; i++) d[i] = 0;
    return 1;
  }
  return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}






// ARRAY STRUCT FUNCTIONS  ------ BORROWED FROM PROF. HERBORDT


array_ptr new_array(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  array_ptr result = (array_ptr) malloc(sizeof(array_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len, sizeof(data_t));
    if (!data) {
      free((void *) result);
      return NULL;  /* Couldn't allocate storage */
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Retrieve array element and store at dest.
   Return 0 (out of bounds) or 1 (successful)
*/
int get_array_element(array_ptr v, long int index, data_t *dest)
{
  if (index < 0 || index >= v->len) {
    return 0;
  }
  *dest = v->data[index];
  return 1;
}

/* Return length of array */
long int get_array_length(array_ptr v)
{
  return v->len;
}

/* initialize an array */
int init_array(array_ptr v)
{
  long int i;

  if (v->len > 0) {
    for (i = 0; i < v->len; i++) {
      v->data[i] = (data_t)(i+1);
    }
    return 1;
  }
  else return 0;
}

int init_array_rand(array_ptr v, double low, double high)
{
  long int i;

  if (v->len > 0) {
    for (i = 0; i < v->len; i++) {
      v->data[i] = (data_t)(fRand(low, high));
    }
    return 1;
  }
  else return 0;
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}

int zero_array(array_ptr m)
{
  if (m->len > 0) {
    long int len = m->len;
    data_t* restrict d = m->data;
    __m256 zv = _mm256_setzero_ps();
    long int i;
    for (i = 0; i <= len - AVX_STRIDE; i += AVX_STRIDE) {
      _mm256_storeu_ps(&d[i], zv);
    }
    for (; i < len; i++) d[i] = 0;
    return 1;
  }
  return 0;
}



// DATASET FUNC DECLARATIONS


dataset_ptr new_dataset(long int len, long int image_len)
{
  long int i;

  /* Allocate and declare header structure */
  dataset_ptr result = (dataset_ptr) malloc(sizeof(dataset));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;
  result->image_len = image_len;

  /* Allocate and declare array */
  if (len > 0) {
    int* nums = (int*) calloc(len, sizeof(int));   // allocate array for labels
    if (!nums) {
      free((void *) result);
      return NULL;  /* Couldn't allocate storage */
    }

    result->nums = nums;

    data_t* data = (data_t *) calloc(len*image_len, sizeof(data_t));   // allocate array for pixel values
    if (!data) {
      free((void *) result);
      return NULL;  /* Couldn't allocate storage */
    }

    result->image_arr = data;
  }
  else result->image_arr = NULL;

  return result;
}

void init_dataset_rand(dataset_ptr d, double low, double high) {

  long int i, j;

  long int len = d->len;
  long int image_len = d->image_len;

  for (i = 0; i < len; i++) {
    d->nums[i] = (int)fRand(0, 10);
  }

  for (i = 0; i < len; i++) {
    for (j = 0; j < image_len; j++) {
      d->image_arr[i*image_len + j] = (data_t)fRand(low, high);
    }
  }

}


void copyImageToInput(dataset_ptr d, array_ptr v, long int index) {
  long int image_len = d->image_len;
  data_t* restrict src = &d->image_arr[index * image_len];
  data_t* restrict dst = v->data;

  long int i;
  for (i = 0; i <= image_len - AVX_STRIDE; i += AVX_STRIDE) {
    _mm256_storeu_ps(&dst[i], _mm256_loadu_ps(&src[i]));
  }
  for (; i < image_len; i++) {
    dst[i] = src[i];
  }
}