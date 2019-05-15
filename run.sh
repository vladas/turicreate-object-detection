#!/bin/bash

echo "Current working directory:"
pwd

echo "Downloading training data..."
./download_training_data.sh

echo "ls -la /storage/"
ls -la /storage/

echo "Train model..."
python train.py

echo "Reduce model size..."
python shrink_model.py