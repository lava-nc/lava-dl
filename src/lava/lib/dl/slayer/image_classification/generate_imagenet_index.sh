#!/bin/bash

# Set the path to your ImageNet data directory
IMAGENET_DIR="/nas-data/pweidel/datasets/imagenet/val"

# Set the output file name
INDEX_FILE="index_file.txt"

# Find all .JPEG files and save their paths to the index file
# The 'find' command is used for recursively searching the files
find "$IMAGENET_DIR" -type f -name "*.JPEG" > "$INDEX_FILE"

echo "Index file generated at $INDEX_FILE"