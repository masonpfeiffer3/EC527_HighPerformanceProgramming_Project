#include <stdio.h>
#include "mnist_loader.h"

int main() {
    dataset train;
    load_mnist(&train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

    printf("Loaded %ld images\n", train.len);

    // Print first 3 images
    for (int i = 0; i < 3; i++) {
        printf("\nLabel: %d\n", train.nums[i]);
        for (int j = 0; j < IMAGE_SIZE; j++) {
            printf("%c", train.image_arr[i * IMAGE_SIZE + j] > 0.5f ? '#' : ' ');
            if ((j + 1) % 28 == 0) printf("\n");
        }
    }

    free_mnist(&train);
    return 0;
}