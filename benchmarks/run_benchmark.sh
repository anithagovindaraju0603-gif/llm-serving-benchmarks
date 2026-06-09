#!/bin/bash

# Wrapper around vLLM's benchmark_serving.py (originally from vLLM repo)
# Works against all four servers since they all expose OpenAI-compatible APIs
#
# Usage:
#   bash run_benchmark.sh baseline    → benchmarks port 8001
#   bash run_benchmark.sh vllm        → benchmarks port 8002
#   bash run_benchmark.sh trtllm      → benchmarks port 8003
#   bash run_benchmark.sh sglang      → benchmarks port 8004

FRAMEWORK=$1

# set port based on framework argument
if [ "$FRAMEWORK" == "baseline" ]; then
    PORT=8001
elif [ "$FRAMEWORK" == "vllm" ]; then
    PORT=8002
elif [ "$FRAMEWORK" == "trtllm" ]; then
    PORT=8003
elif [ "$FRAMEWORK" == "sglang" ]; then
    PORT=8004
else
    echo "Usage: bash run_benchmark.sh [baseline|vllm|trtllm|sglang]"
    exit 1
fi

# request rates to sweep. starts easy, ramps up until server breaks
REQUEST_RATES=(1 2 4 8 16 32)

for RATE in "${REQUEST_RATES[@]}"
do
    echo "Running benchmark: framework=$FRAMEWORK rate=$RATE req/sec"

    python benchmark_serving.py \
        --backend openai \
        --base-url http://localhost:$PORT \
        --dataset-name sharegpt \
        --dataset-path /workspace/benchmarks/ShareGPT_V3.json \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --num-prompts 1000 \
        --request-rate $RATE \
        --save-result \
        --result-dir /workspace/benchmarks/results/$FRAMEWORK \
        --result-filename ${FRAMEWORK}_rate${RATE}.json

    echo "Done. Results saved to /workspace/benchmarks/results/$FRAMEWORK/${FRAMEWORK}_rate${RATE}.json"
done

echo "All benchmarks complete for $FRAMEWORK"

#ShareGPT was a website where people shared their real ChatGPT conversations. Someone scraped it and released it as a dataset.
#It's used here because it has real user prompts, which is more realistic than synthetic prompts. It also has a variety of prompt types and lengths, which is good for testing the servers under different conditions.

#Things that we measure in the benchmark:
# TTFT: time from sending request to receiving FIRST token back. This is what the user actually feels as "lag"
# TPOT: time between each subsequent token, this is the streaming speed after first token
# E2E latency: total time from request sent to last token received
# tokens/sec: how many output tokens per second across all requests