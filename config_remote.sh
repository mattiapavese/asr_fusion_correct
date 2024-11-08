#!/bin/bash
apt update -y && apt upgrade -y

python3.11 -m pip install transformers datasets librosa pydub python-box protobuf sentencepiece peft

#if you want to run also prepare_dataset.py, you need polars
#python3.11 -m install polars

#if not running on runpod with pyTorch already installed, install torch
#python3.11 -m install torch

apt install -y ffmpeg

# to push on remote, remember to set up github access token with 
#  export GITHUB_ACCESS_TOKEN=<your_token>
# and eventually 
#  git remote set-url origin https://$GITHUB_ACCESS_TOKEN@github.com/mattiapavese/asr_fusion_correct.git

