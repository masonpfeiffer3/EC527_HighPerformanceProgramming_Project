.PHONY: data run clean

CC = gcc #compiler

SRC = src/mnist_loader.c src/main.c \
      src/model.c \
      src/array_matrix_funcs.c \
      src/array_matrix_math.c \
      src/kernels.c
HDR = src/mnist_loader.h \
      src/model.h \
      src/array_matrix_funcs.h \
      src/array_matrix_math.h \
      src/kernels.h
TARGET = mnist_c #executable name

all: $(TARGET)
#currently avoiding any optimizations by the compiler
$(TARGET): $(SRC) $(HDR)
	$(CC) -O3 -o $(TARGET) $(SRC) -lm -fopenmp -mavx2 -mfma

data:
	bash scripts/download_data.sh

run: all data
	./$(TARGET)

clean:
	rm -f $(TARGET)