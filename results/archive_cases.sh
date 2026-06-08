#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <comment>"
    echo "Compression failed."
    exit 1
fi

comment=$1
current_time=$(date +"%Y%m%d_%H%M%S")
zip -r "./archives/cases_redundant_${current_time}_${comment}.zip" ./cases/

if [ $? -eq 0 ]; then
    echo "Compression successful. Files not removed."
else
    echo "Compression failed."
fi
