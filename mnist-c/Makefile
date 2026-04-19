.PHONY: data run clean

CC = gcc #compiler

SRC = src/mnist_loader.c src/main.c #sources
TARGET = mnist_c #executable name

all: $(TARGET)
#currently avoiding any optimizations by the compiler
$(TARGET): $(SRC)
	$(CC) -O0 -o $(TARGET) $(SRC)

data:
	bash scripts/download_data.sh

run: all data
	./$(TARGET)

clean:
	rm -f $(TARGET)