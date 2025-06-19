#!/bin/bash

# -----------------------------
# System dependencies (via apt)
# -----------------------------
echo "🔧 Installing system packages..."
apt-get update
apt-get install -y sox libsndfile1 ffmpeg libsox-fmt-mp3 jq

# -----------------------------
# Python packages (via pip)
# -----------------------------
echo "🐍 Installing Python packages..."
pip install wheel
pip install wget
pip install text-unidecode
pip install matplotlib>=3.3.2
pip install Cython
pip install transformers==4.40.0
pip install huggingface_hub==0.21.3
pip install "tokenizers>=0.21"
pip install datasets==2.19.0
pip install numpy==1.26.4
pip install scipy==1.12.0
pip install scikit-learn==1.2.2

# -----------------------------
# NVIDIA NeMo Toolkit
# -----------------------------
echo "🧠 Installing NVIDIA NeMo v1.23.0..."
pip install git+https://github.com/NVIDIA/NeMo.git@v1.23.0#egg=nemo_toolkit[all]

echo "✅ Setup complete."
