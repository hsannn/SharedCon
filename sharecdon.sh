#!/bin/bash 
 
echo "### START DATE=$(date)" 
echo "### HOSTNAME=$(hostname)" 
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" 
 
# conda 환경 활성화. 
source  ~/.bashrc 
conda activate shacon
 
# cuda 11.0 환경 구성. 
ml unload cuda/11.2 nccl/2.8.4/cuda11.2 
ml load cuda/11.0 nccl/2.8.4/cuda11.0 
ml list 
 
cd /home/sunny5574/new_shacon
 
# 활성화된 환경에서 코드 실행. 
python -u eval.py



echo "###" 
echo "### END DATE=$(date)"