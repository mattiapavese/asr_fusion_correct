#!/bin/bash
#to be used from local machine to remote to transfer dataset files
$remote_ip="194.68.245.79" #replace with ip of machine where to transfer
scp -i ~/.ssh/id_ed25519 ./dataset.tsv  root@$remote_ip:/workspace/asr_fusion_correct/
scp -i ~/.ssh/id_ed25519 -r  ./cv-corpus-19.0-2024-09-13  root@1$remote_ip:/workspace/asr_fusion_correct/