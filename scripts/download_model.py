import os
from huggingface_hub import login, snapshot_download

def download_model():
    hf_token = os.environ.get('hf_token')
    if not hf_token:
        raise ValueError("hf_token not found in environment")
    
    login(token=hf_token)

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    save_path = "/workspace/models/Meta-Llama-3-8B-Instruct"

    print(f"Downloading {model_id} to {save_path}...")
    
    #The "snapshot" in the name means it downloads the repo exactly as it exists on HuggingFace at that moment. Just the weights alone aren't enough,it needs the config to know the architecture, and the tokenizer to process text. 
    snapshot_download(
        repo_id=model_id,
        local_dir=save_path,
        local_dir_use_symlinks=False
    )
    
    print("Done!")

if __name__ == "__main__":
    download_model()