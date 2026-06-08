#!/bin/bash

wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv
mv ./AzureLLMInferenceTrace_code_1week.csv ./workloads/Azure/Azure_code.csv
wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv
mv ./AzureLLMInferenceTrace_conv_1week.csv ./workloads/Azure/Azure_conv.csv