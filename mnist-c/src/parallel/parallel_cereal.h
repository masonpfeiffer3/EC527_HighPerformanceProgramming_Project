#ifndef PARALLEL_CEREAL_H
#define PARALLEL_CEREAL_H

#include "../params.h"
#include "../serial/array_matrix_funcs.h"

// Entry point: trains and tests the network using OpenMP batched parallelism.
// num_threads: how many OpenMP threads to use (must evenly divide BATCH_SIZE=100)
void parallel_MNIST(dataset_ptr train_data, dataset_ptr test_data, int num_threads);

#endif
