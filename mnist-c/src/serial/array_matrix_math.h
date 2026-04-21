#ifndef ARRAY_MATRIX_MATH_H
#define ARRAY_MATRIX_MATH_H

// Forward declarations (these types are defined in params.h)
typedef struct array_struct *array_ptr;
typedef struct matrix_struct *matrix_ptr;

// Matrix-vector operations
int matrix_vector_mult(matrix_ptr m, array_ptr v, array_ptr v_out);

// Matrix-matrix operations
int matrix_matrix_add(matrix_ptr m1, matrix_ptr m2, matrix_ptr m_out);
int matrix_transpose(matrix_ptr m, matrix_ptr m_out)

// Vector-vector operations
int vector_vector_add(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_vector_sub(array_ptr v1, array_ptr v2, array_ptr v_out);
int vector_vector_mult(array_ptr v1, array_ptr v2, matrix_ptr v_out);
int vector_vector_elementwise_mult(array_ptr v1, array_ptr v2, array_ptr v_out);

// Utility
int vector_copy(array_ptr source, array_ptr dest);
array_ptr numToVec(int num);

#endif