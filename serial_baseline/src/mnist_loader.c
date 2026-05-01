#include <stdio.h>
#include <stdlib.h>
#include "mnist_loader.h"

#define IMAGE_MAGIC 2051
#define LABEL_MAGIC 2049

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void load_mnist(dataset_ptr d, const char *image_file, const char *label_file) {
    FILE *image_fp = fopen(image_file, "rb");
    FILE *label_fp = fopen(label_file, "rb");
    if (image_fp == NULL || label_fp == NULL) {
        fprintf(stderr, "Error opening files.\n");
        exit(EXIT_FAILURE);
    }

    int image_magic, label_magic;
    fread(&image_magic, sizeof(int), 1, image_fp);
    fread(&label_magic, sizeof(int), 1, label_fp);
    image_magic = reverse_int(image_magic);
    label_magic = reverse_int(label_magic);
    if (image_magic != IMAGE_MAGIC || label_magic != LABEL_MAGIC) {
        fprintf(stderr, "Invalid MNIST files.\n");
        exit(EXIT_FAILURE);
    }

    int num_images, num_labels;
    fread(&num_images, sizeof(int), 1, image_fp);
    fread(&num_labels, sizeof(int), 1, label_fp);
    num_images = reverse_int(num_images);
    num_labels = reverse_int(num_labels);
    if (num_images != num_labels) {
        fprintf(stderr, "Number of images and labels do not match.\n");
        exit(EXIT_FAILURE);
    }

    int rows, cols;
    fread(&rows, sizeof(int), 1, image_fp);
    fread(&cols, sizeof(int), 1, image_fp);
    if (reverse_int(rows) != 28 || reverse_int(cols) != 28) {
        fprintf(stderr, "Unexpected image dimensions.\n");
        exit(EXIT_FAILURE);
    }

    d->len = num_images;
    d->image_len = IMAGE_SIZE;
    d->nums = (int *)malloc(num_labels * sizeof(int));
    d->image_arr = (float *)malloc(num_images * IMAGE_SIZE * sizeof(float));

    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        fread(&label, sizeof(unsigned char), 1, label_fp);
        d->nums[i] = (int)label;
    }

    unsigned char buffer[IMAGE_SIZE];
    for (int i = 0; i < num_images; i++) {
        fread(buffer, sizeof(unsigned char), IMAGE_SIZE, image_fp);
        for (int j = 0; j < IMAGE_SIZE; j++)
            d->image_arr[i * IMAGE_SIZE + j] = (float)buffer[j] / 255.0f;
    }

    fclose(image_fp);
    fclose(label_fp);
}

void free_mnist(dataset_ptr d) {
    free(d->nums);
    free(d->image_arr);
    d->nums = NULL;
    d->image_arr = NULL;
    d->len = 0;
    d->image_len = 0;
}