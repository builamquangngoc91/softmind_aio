from transformers import AutoModelForCausalLM, AutoTokenizer


import os

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

from datasets import load_dataset
dataset = load_dataset("thailevann/vlsp_legal_pretrain")

print(f"Number of samples: {len(dataset['train'])}")