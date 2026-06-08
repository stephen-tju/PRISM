#!/bin/bash

mkdir -p datasets/ShareGPT
mkdir -p datasets/LMSYS-Chat-1M

mkdir -p workloads/BurstGPT
mkdir -p workloads/Azure

# Download ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
mv ./ShareGPT_V3_unfiltered_cleaned_split.json ./datasets/ShareGPT/ShareGPT.json

# Download LMSYS-Chat-1M dataset
# to-do

# Download BurstGPT workload
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv
mv ./BurstGPT_1.csv ./workloads/BurstGPT/BurstGPT.csv

# Download Azure workload
# to-do
