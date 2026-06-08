#!/bin/bash

mkdir -p datasets/ShareGPT
# mkdir -p datasets/LMSYS-Chat-1M

# mkdir -p workloads/BurstGPT
mkdir -p workloads/Azure
mkdir -p workloads/Azure_code
mkdir -p workloads/Azure_conv


wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
mv ./ShareGPT_V3_unfiltered_cleaned_split.json ./datasets/ShareGPT/ShareGPT.json

wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv
mv ./AzureLLMInferenceTrace_code_1week.csv ./workloads/Azure/Azure_code.csv
wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv
mv ./AzureLLMInferenceTrace_conv_1week.csv ./workloads/Azure/Azure_conv.csv

# wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv
# mv ./BurstGPT_1.csv ./workloads/BurstGPT/BurstGPT.csv


python preprocess_workloads.py
