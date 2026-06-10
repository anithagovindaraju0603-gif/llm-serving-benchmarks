from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from pydantic import BaseModel
import torch
import json

app = FastAPI()

MODEL_PATH = "/workspace/models/Meta-Llama-3-8B-Instruct"

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
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_tokens,
        do_sample=False,
        streamer=streamer
    )

    # run generation in background thread so we can stream tokens as they arrive
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate_stream():
        generated_tokens = 0
        for new_text in streamer:
            generated_tokens += len(tokenizer.encode(new_text))
            chunk = {
                "id": "cmpl-baseline",
                "object": "text_completion",
                "model": req.model,
                "choices": [
                    {
                        "text": new_text,
                        "index": 0,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # final chunk with finish_reason
        final_chunk = {
            "id": "cmpl-baseline",
            "object": "text_completion",
            "model": req.model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": generated_tokens,
                "total_tokens": inputs["input_ids"].shape[1] + generated_tokens
            }
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}