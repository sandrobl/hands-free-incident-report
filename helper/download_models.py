# scripts/download_models.py
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_id = "google/gemma-4-e2b-it"
cache_dir = "/workspace/gemma-4"

print("Downloading model...")
AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
)
AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
print("Done.")