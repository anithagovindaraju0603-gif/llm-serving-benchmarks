from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = FastAPI()

MODEL_PATH = "/models/Meta-Llama-3-8B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded.")


class CompletionRequest(BaseModel):
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt: str
    max_tokens: int = 256

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    start = time.time()

    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end = time.time()

    return {
        "id": "cmpl-baseline",
        "object": "text_completion",
        "model": req.model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": outputs.shape[1]
        },
        "latency_seconds": round(end - start, 3)
    }

@app.get("/health")
def health():
    return {"status": "ok"}