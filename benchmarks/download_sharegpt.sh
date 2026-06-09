#!/bin/bash

# Download ShareGPT dataset — the prompts used for benchmarking

echo "Downloading ShareGPT dataset..."

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
    -O /workspace/benchmarks/ShareGPT_V3.json

echo "Done. Saved to /workspace/benchmarks/ShareGPT_V3.json"