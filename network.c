#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define I_SIZE 784
#define H0_SIZE 16
#define H1_SIZE 16
#define L_SIZE 10

#define INIT_LOW 0
#define INIT_HIGH 10

typedef struct {
  long int rowlen;
  long int collen;
  data_t *data;
} matrix_rec, *matrix_ptr;

typedef struct {
  long int len;
  data_t *data;
} array_rec, *array_ptr;


// array func declarations
array_ptr new_array(long int len);
int get_array_element(array_ptr v, long int index, data_t *dest);
long int get_array_length(array_ptr v);
int set_array_length(array_ptr v, long int index);
int init_array(array_ptr v, long int len);

// matrix func declarations
double fRand(double fMin, double fMax);
matrix_ptr new_matrix(long int len);
int set_matrix_rowlen(matrix_ptr m, long int index);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m, long int len);
int zero_matrix(matrix_ptr m, long int len);
int init_matrix_rand(matrix_ptr m, long int rowlen);
int init_matrix_rand_ptr(matrix_ptr m, long int rowlen);





int main() {

    // INPUT ARRAY
    array_ptr input = new_array(I_SIZE);
    
    // H0 ARRAY, WEIGHTS, and BIASES
    array_ptr H0 = new_array(H0_SIZE);
    matrix_ptr H0_W = new_matrix(H0_SIZE)

    // H1 ARRAY
    array_ptr H1 = new_array(H1_SIZE);


    // OUTPUT ARRAY
    array_ptr output = new_array(L_SIZE);



    return 0;
}








// MATRIX STRUCT FUNCTIONS ----- BORROWED FROM PROF. HERBORDT


matrix_ptr new_matrix(long int rowlen, long int collen)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->rowlen = rowlen;

  /* Allocate and declare array */
  if (rowlen > 0 && collen > 0) {
    data_t *data = (data_t *) calloc(rowlen*collen, sizeof(data_t));
    if (!data) {
      free((void *) result);
      printf("COULDN'T ALLOCATE %ld BYTES STORAGE \n",
                                  rowlen * rowlen * sizeof(data_t) );
      exit(-1);
    }
    result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Return row length of matrix */
long int get_matrix_rowlen(matrix_ptr m)
{
  return m->rowlen;
}

/* Return row length of matrix */
long int get_matrix_collen(matrix_ptr m)
{
  return m->collen;
}


/* initialize matrix */
int init_matrix(matrix_ptr m)
{
  long int i;

  if (m->rowlen > 0 && m->collen > 0) {
    for (i = 0; i < m->rowlen*m->collen; i++) {
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
int init_matrix_rand(matrix_ptr m)
{
  long int i;

  if (m->rowlen > 0 && m->collen > 0) {
    for (i = 0; i < m->rowlen*m->collen; i++) {
      m->data[i] = (data_t)(fRand(INIT_LOW, INIT_HIGH));
    }
    return 1;
  }
  else return 0;
}

int zero_matrix(matrix_ptr m)
{
  long int i;

  if (m->rowlen > 0 && m->collen > 0) {
    for (i = 0; i < m->rowlen*m->collen; i++) {
      m->data[i] = 0;
    }
    return 1;
  }
  else return 0;
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

/* Set length of array */
int set_array_length(array_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* initialize an array */
int init_array(array_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len; i++) {
      v->data[i] = (data_t)(i+1);
    }
    return 1;
  }
  else return 0;
}

data_t *get_array_start(array_ptr v)
{
  return v->data;
}



// MVM CODE (TO BE OPTIMIZED)

matrix_vector_mult()