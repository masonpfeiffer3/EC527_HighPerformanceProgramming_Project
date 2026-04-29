.PHONY: data run clean

CC     = gcc
CFLAGS = -O0 -fopenmp

SRC = src/mnist_loader.c src/main.c \
      src/serial/array_matrix_funcs.c src/serial/array_matrix_math.c \
      src/parallel/parallel_cereal.c src/parallel/parallel_kernels.c

HDR = src/mnist_loader.h src/params.h \
      src/serial/array_matrix_funcs.h src/serial/array_matrix_math.h \
      src/parallel/parallel_cereal.h src/parallel/parallel_kernels.h

TARGET = mnist_c

all: $(TARGET)

# No compiler optimizations -- baseline for measuring algorithmic restructuring only
$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

data:
	bash scripts/download_data.sh

# Example: make run ARGS="10"   (10 threads, 10 samples/thread)
run: all data
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)
