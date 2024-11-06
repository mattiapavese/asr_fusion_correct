apt update -y && apt upgrade -y

python3.11 -m venv venv
venv/bin/pip install transformers datasets librosa pydub python-box torch protobuf sentencepiece

#if you want to run also prepare_dataset.py, you need polars
#venv/bin/pip install polars

apt install -y ffmpeg

# to push on remote, remember to set up github access token with 
# GITHUB_ACCESS_TOKEN=<your_token>
