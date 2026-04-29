#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "parallel/parallel_cereal.h"
#include "params.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        fprintf(stderr, "  num_threads must evenly divide BATCH_SIZE (100)\n");
        return 1;
    }
    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) {
        fprintf(stderr, "Error: num_threads must be a positive integer\n");
        return 1;
    }

    dataset_ptr train_data = new_dataset(TRAIN_SIZE, IMAGE_SIZE);
    dataset_ptr test_data  = new_dataset(TEST_SIZE,  IMAGE_SIZE);
    load_mnist(train_data, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    load_mnist(test_data,  "data/t10k-images-idx3-ubyte",  "data/t10k-labels-idx1-ubyte");

    struct timespec prog_start, prog_end;
    clock_gettime(CLOCK_MONOTONIC, &prog_start);
    parallel_MNIST(train_data, test_data, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &prog_end);
    printf("Total: %.4f s\n",
           (prog_end.tv_sec  - prog_start.tv_sec) +
           (prog_end.tv_nsec - prog_start.tv_nsec) / 1e9);

    free_mnist(train_data);
    free_mnist(test_data);
    return 0;
}
