#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/emotion_classification.py

# deactive the venv
deactivate