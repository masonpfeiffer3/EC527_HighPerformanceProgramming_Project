#ifndef ARRAY_MATRIX_FUNCS_H
#define ARRAY_MATRIX_FUNCS_H

typedef double data_t;

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
int init_array(array_ptr v);

// matrix func declarations
double fRand(double fMin, double fMax);
matrix_ptr new_matrix(long int rowlen, long int collen);
long int get_matrix_rowlen(matrix_ptr m);
long int get_matrix_collen(matrix_ptr m);
int init_matrix(matrix_ptr m);
int zero_matrix(matrix_ptr m);
int init_matrix_rand(matrix_ptr m);


#endif