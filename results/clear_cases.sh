#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <comment>"
    echo "Compression failed. Files not removed."
    exit 1
fi

comment=$1
current_time=$(date +"%Y%m%d_%H%M%S")
zip -r "./archives/cases_backup_${current_time}_${comment}.zip" ./cases/


if [ $? -eq 0 ]; then
    echo "Compression successful. Removing files..."
    rm -rf ./cases/*
else
    echo "Compression failed. Files not removed."
fi
