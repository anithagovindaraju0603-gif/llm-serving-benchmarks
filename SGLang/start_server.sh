#!/bin/bash

docker run --gpus all \
    -v /workspace/models:/models \
    -p 8004:8000 \
    lmsysorg/sglang:latest \
    python -m sglang.launch_server \
    --model-path /models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-total-tokens 4096 \
    --max-num-reqs 256