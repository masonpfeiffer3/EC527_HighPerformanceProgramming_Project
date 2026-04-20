#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "params.h"

void load_mnist(dataset_ptr d, const char *image_file, const char *label_file);
void free_mnist(dataset_ptr d);

#endif