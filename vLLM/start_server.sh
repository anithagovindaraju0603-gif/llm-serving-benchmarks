#!/bin/bash

#no custom Dockerfile needed since we use their pre-built image
docker run --gpus all \
    -v /workspace/models:/models \
    -p 8002:8000 \
    vllm/vllm-openai:latest \
    --model /models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \

    #max context window, caps KV cache memory usage
    --max-model-len 4096 \

    #vLLM can use 90% of GPU memory
    --gpu-memory-utilization 0.9 \

    #at most 256 requests generating simultaneously!
    --max-num-seqs 256