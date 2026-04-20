#include <stdio.h>
#include "mnist_loader.h"
#include "serial/cereal.h"
#include "params.h"

int main() {
    dataset_ptr train_data = new_dataset(TRAIN_SIZE, IMAGE_SIZE);
    dataset_ptr test_data = new_dataset(TEST_SIZE, IMAGE_SIZE);
    load_mnist(train_data, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    load_mnist(test_data, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");


    serial_MNIST(train_data, test_data);


    // printf("Loaded %ld images\n", train.len);

    // // Print first 3 images
    // for (int i = 0; i < 3; i++) {
    //     printf("\nLabel: %d\n", train.nums[i]);
    //     for (int j = 0; j < IMAGE_SIZE; j++) {
    //         printf("%c", train.image_arr[i * IMAGE_SIZE + j] > 0.5f ? '#' : ' ');
    //         if ((j + 1) % 28 == 0) printf("\n");
    //     }
    // }

    free_mnist(train_data);
    free_mnist(test_data);
    return 0;
}