#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#define IMAGE_SIZE 784
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

typedef struct{
    long int len; //elements in dataset
    long int image_len; //always 784 for MNIST
    int* nums; //labels for each image, 0-9
    float* image_arr; //flattened array of all images
} dataset, *dataset_ptr;

void load_mnist(dataset_ptr d, const char *image_file, const char *label_file);
void free_mnist(dataset_ptr d);

#endif