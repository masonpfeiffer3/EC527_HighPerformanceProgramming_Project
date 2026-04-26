.PHONY: data run clean

CC = gcc #compiler

SRC = src/mnist_loader.c src/main.c src/serial/cereal.c src/serial/array_matrix_funcs.c src/serial/array_matrix_math.c src/serial/kernels.c #sources
HDR = src/mnist_loader.h src/serial/cereal.h src/serial/array_matrix_funcs.h src/serial/array_matrix_math.h src/serial/kernels.h #headers
TARGET = mnist_c #executable name

all: $(TARGET)
#currently avoiding any optimizations by the compiler
$(TARGET): $(SRC) $(HDR)
	$(CC) -O0 -o $(TARGET) $(SRC) -lm -fopenmp

data:
	bash scripts/download_data.sh

run: all data
	for t in 1 2 4 5 6 7 8 9 10 11 12 16 20 24 32; do \
		echo "=== Threads: $$t ==="; \
		OMP_NUM_THREADS=$$t ./$(TARGET); \
	done

clean:
	rm -f $(TARGET)