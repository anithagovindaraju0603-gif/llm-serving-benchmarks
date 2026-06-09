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


class Request(BaseModel):
    prompt: str
    max_tokens: int = 256


@app.post("/generate")
def generate(req: Request):
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
        "text": text,
        "latency_seconds": round(end - start, 3)
    }


@app.get("/health")
def health():
    return {"status": "ok"}