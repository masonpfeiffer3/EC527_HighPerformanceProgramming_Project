#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
mkdir -p $DATA_DIR
BASE_URL="https://raw.githubusercontent.com/fgnt/mnist/master"

if [ ! -f $DATA_DIR/train-images-idx3-ubyte ]; then
    echo "Downloading training images..."
    curl -o $DATA_DIR/train-images-idx3-ubyte.gz $BASE_URL/train-images-idx3-ubyte.gz
    gunzip $DATA_DIR/train-images-idx3-ubyte.gz
fi

if [ ! -f $DATA_DIR/train-labels-idx1-ubyte ]; then
    echo "Downloading training labels..."
    curl -o $DATA_DIR/train-labels-idx1-ubyte.gz $BASE_URL/train-labels-idx1-ubyte.gz
    gunzip $DATA_DIR/train-labels-idx1-ubyte.gz
fi

if [ ! -f $DATA_DIR/t10k-images-idx3-ubyte ]; then
    echo "Downloading test images..."
    curl -o $DATA_DIR/t10k-images-idx3-ubyte.gz $BASE_URL/t10k-images-idx3-ubyte.gz
    gunzip $DATA_DIR/t10k-images-idx3-ubyte.gz
fi

if [ ! -f $DATA_DIR/t10k-labels-idx1-ubyte ]; then
    echo "Downloading test labels..."
    curl -o $DATA_DIR/t10k-labels-idx1-ubyte.gz $BASE_URL/t10k-labels-idx1-ubyte.gz
    gunzip $DATA_DIR/t10k-labels-idx1-ubyte.gz
fi

echo "Done! Files are in $DATA_DIR"