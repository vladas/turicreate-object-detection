#!/bin/bash

echo "\n> Current working directory:"
pwd

echo "\n> Install dependencies..."
./install.sh

echo "\n> Downloading training data..."
./download_training_data.sh

echo "ls -la /storage/"
ls -la /storage/

echo "\n> Train model..."
python train.py

echo "\n> Reduce model size..."
python shrink_model.py