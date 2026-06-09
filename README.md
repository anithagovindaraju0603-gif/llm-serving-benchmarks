# LLM Serving Benchmarks

Benchmarking LLaMA-3-8B-Instruct across four serving frameworks:
- Vanilla HuggingFace + FastAPI (baseline)
- vLLM
- TensorRT-LLM
- SGLang

Measuring throughput, latency (p50/p95/p99), TTFT, and TPOT under realistic load using ShareGPT prompts.

---

## Repo Structure