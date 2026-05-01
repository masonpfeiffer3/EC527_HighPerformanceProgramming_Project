.PHONY: data run clean

CC = gcc #compiler

SRC = src/mnist_loader.c src/main.c src/cereal.c src/array_matrix_funcs.c src/array_matrix_math.c src/kernels.c #sources
HDR = src/mnist_loader.h src/cereal.h src/array_matrix_funcs.h src/array_matrix_math.h src/kernels.h #headers
TARGET = serial_baseline #executable name

all: $(TARGET)
#currently avoiding any optimizations by the compiler
$(TARGET): $(SRC) $(HDR)
	$(CC) -O0 -o $(TARGET) $(SRC) -lm

data:
	bash scripts/download_data.sh

run: all data
	./$(TARGET)

clean:
	rm -f $(TARGET)