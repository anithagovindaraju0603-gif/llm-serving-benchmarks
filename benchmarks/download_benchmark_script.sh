#!/bin/bash

# Download benchmark_serving.py from vLLM repo, since vLLM updates it regularly as they add features

echo "Downloading benchmark_serving.py from vLLM repo..."

wget https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/benchmark_serving.py \
    -O /workspace/benchmarks/benchmark_serving.py

echo "Done. Saved to /workspace/benchmarks/benchmark_serving.py"

#benchmark_serving.py is a script that fires 1000 real prompts at the 4 servers, measures how fast it responds at different load levels, and saves the numbers so we can compare all four frameworks.
#Inside benchmark_serving.py:
# 1. Opens ShareGPT_V3.json
#    picks 1000 conversations
#    extracts the human prompts

# 2. Starts a timer

# 3. Sends requests to your server at the rate you specify
#    if rate=8, sends one request every 125ms
#    each request is a real ShareGPT prompt

# 4. For each request it records:
#    exactly when it was sent
#    exactly when the first token came back  (TTFT)
#    exactly when the last token came back   (E2E latency)

# 5. After all 1000 requests done:
#    calculates p50, p95, p99 for all metrics
#    saves everything to a JSON file