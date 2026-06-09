#!/bin/bash

# Start TensorRT-LLM server via Triton Inference Server

# ABout TRITON
# TRT-LLM builds the engine but it doesn't have its own HTTP server
# Triton is NVIDIA's serving framework it loads the TRT-LLM engine and exposes an HTTP API same idea as FastAPI for our baseline

docker run --gpus all \
    -v /workspace/models:/models \
    -v /workspace/trt-engine:/engine \
    -p 8003:8000 \
    nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 \
    bash -c "
        # copy the engine into triton's model repository format
        cp -r /engine/trt-engine /opt/tritonserver/model_repo/llama/

        # start triton server
        tritonserver \
            --model-repository /opt/tritonserver/model_repo \
            --http-port 8000 \
            --grpc-port 8001 \
            --metrics-port 8002
    "

# TRT-LLM server:
# Triton takes the TRT-LLM engine and exposes it via HTTP, and then client sends request to 
# Triton  → engine → response back to client