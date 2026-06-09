#!/bin/bash

# TensorRT-LLM engine build script

# Step 1 — convert_checkpoint.py
# → HuggingFace weights are in their own format
# → TRT-LLM can't read them directly
# → this script converts them to TRT-LLM's format

# Step 2 — trtllm-build
# → takes the checkpoint and compiles it into a hardware-specific engine
# → fuses kernels, optimizes for my exact GPU(A100)

docker run --gpus all \
    -v /workspace/models:/models \
    -v /workspace/trt-engine:/engine \
    nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 \
    bash -c "
        # Step 1 — convert HuggingFace weights to TRT-LLM checkpoint format
        python /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
            --model_dir /models/Meta-Llama-3-8B-Instruct \
            --output_dir /engine/trt-ckpt \
            --dtype float16

        # Step 2 — compile the checkpoint into a TensorRT engine
        trtllm-build \
            --checkpoint_dir /engine/trt-ckpt \
            --output_dir /engine/trt-engine \
            --gemm_plugin float16 \
            --gpt_attention_plugin float16 \
            --max_batch_size 32 \
            --max_input_len 2048 \
            --max_output_len 1024 \
            --use_paged_context_fmha enable
    "

echo "Engine build complete. Saved to /workspace/trt-engine"

#Change any of the above params, must rebuild the entire engine!! Less flexible than vLLM