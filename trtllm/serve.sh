#!/bin/bash

# Start TensorRT-LLM server via Triton Inference Server
# Run build_engine.sh FIRST before this
# Runs on port 8003
#
# About Triton:
# TRT-LLM builds the engine but it doesn't have its own HTTP server
# Triton is NVIDIA's serving framework — loads the TRT-LLM engine and exposes an HTTP API
# Same idea as FastAPI for our baseline
#
# Flow:
# client sends request → Triton → TRT-LLM engine → response back to client
#
# Steps inside container:
# 1. copy compiled engine into Triton's model repository format
# 2. start Triton server

docker run --gpus all \
    -v /workspace/models:/models \
    -v /workspace/trt-engine:/engine \
    -p 8003:8000 \
    nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 \
    bash -c "
        mkdir -p /opt/tritonserver/model_repo/llama \
        && cp -r /engine/trt-engine /opt/tritonserver/model_repo/llama/ \
        && tritonserver \
            --model-repository /opt/tritonserver/model_repo \
            --http-port 8000 \
            --grpc-port 8001 \
            --metrics-port 8002
    "