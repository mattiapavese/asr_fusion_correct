#!/bin/bash
mkdir cv-corpus-19.0-2024-09-13
tar -xzvf ./cv-corpus-19.0-2024-09-13.tar.gz -C ./cv-corpus-19.0-2024-09-13/

# eventually find and delete all files starting with "._" within the corpus dir,
# that are artifacts created by macOS
find "./cv-corpus-19.0-2024-09-13" -type f -name '._*' -exec rm -f {} +