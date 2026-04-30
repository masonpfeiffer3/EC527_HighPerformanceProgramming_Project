#ifndef MODEL_H
#define MODEL_H

#include "array_matrix_funcs.h"

/* Forward-declare the batch scratch struct. */
typedef struct BatchScratch BatchScratch;

/* Per-thread gradient accumulator (same shape as serial ThreadGradSum). */
typedef struct {
  matrix_ptr H0_W_grad_sum, H1_W_grad_sum, L_W_grad_sum;
  array_ptr  H0_B_grad_sum, H1_B_grad_sum, L_B_grad_sum;
} ThreadGradSum;

void parallel_MNIST(dataset_ptr train_data, dataset_ptr test_data);
void train_MNIST(dataset_ptr train_data);
void test_MNIST(dataset_ptr test_data);
void init_MNIST(void);

void feedforward_batch(BatchScratch *s, int actual_S);
void backprop_batch(BatchScratch *s, ThreadGradSum *ts, int actual_S);

#endif
