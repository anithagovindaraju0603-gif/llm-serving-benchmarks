#!/bin/bash

#no custom Dockerfile needed since we use their pre-built image
#max context window, caps KV cache memory usage
#vLLM can use 90% of GPU memory
#at most 256 requests generating simultaneously!
docker run --gpus all \
    -v /workspace/models:/models \
    -p 8002:8000 \
    vllm/vllm-openai:latest \
    --model /models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256