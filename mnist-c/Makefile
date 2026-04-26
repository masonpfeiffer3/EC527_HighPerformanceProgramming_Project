.PHONY: data run clean

CC = gcc #compiler

SRC = src/mnist_loader.c src/main.c src/serial/cereal.c src/serial/array_matrix_funcs.c src/serial/array_matrix_math.c src/serial/kernels.c #sources
HDR = src/mnist_loader.h src/serial/cereal.h src/serial/array_matrix_funcs.h src/serial/array_matrix_math.h src/serial/kernels.h #headers
TARGET = mnist_c #executable name

all: $(TARGET)
#currently avoiding any optimizations by the compiler
$(TARGET): $(SRC) $(HDR)
	$(CC) -O3 -o $(TARGET) $(SRC) -lm

data:
	bash scripts/download_data.sh

run: all data
	./$(TARGET)

clean:
	rm -f $(TARGET)