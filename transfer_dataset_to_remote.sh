#!/bin/bash
#to be used from local machine to remote to transfer dataset files
remote_user="root"
remote_ip="157.157.221.29" #replace with ip of machine where to transfer
remote_path="/workspace/asr_fusion_correct/"
remote_port=16385

scp -i ~/.ssh/id_ed25519 -P $remote_port ./dataset.tsv  $remote_user@$remote_ip:$remote_path

#transfer archive .tar.gz!!
scp -i ~/.ssh/id_ed25519 -r -P $remote_port  ./cv-corpus-19.0-2024-09-13.tar.gz  $remote_user@$remote_ip:$remote_path