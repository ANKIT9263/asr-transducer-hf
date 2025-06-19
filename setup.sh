#!/bin/bash

# Update package list and install system packages
sudo apt-get update
sudo apt-get install -y sox libsndfile1 ffmpeg

# Install Python packages
pip install -r requirements.txt
