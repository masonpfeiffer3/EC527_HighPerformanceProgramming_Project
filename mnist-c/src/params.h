#ifndef PARAMS_H
#define PARAMS_H

#define TRAIN_SIZE 60
#define TEST_SIZE 10000
#define IMAGE_SIZE 784
#define EULER_NUMBER_F 2.71828182846

typedef float data_t;

typedef struct {
    long int rows;
    long int cols;
    data_t *data;
} matrix_rec, *matrix_ptr; //flattened 2D

typedef struct {
    long int len;
    data_t *data;
} array_rec, *array_ptr; //1D

typedef struct {
    long int len;
    long int image_len;
    int *nums;
    data_t *image_arr;
} dataset, *dataset_ptr;


#endif