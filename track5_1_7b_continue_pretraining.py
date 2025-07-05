#!/usr/bin/env python3
"""
Qwen3-1.7B Continual Pretraining Script
Converted from Jupyter notebook for legal corpus training
"""

import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from huggingface_hub import login


def check_gpu():
    """Check GPU availability and specifications"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_model():
    """Load and configure the model"""
    max_seq_length = 2048
    dtype = None  # None for auto detection
    load_in_4bit = True
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )
    
    return model, tokenizer, max_seq_length


def prepare_dataset():
    """Format the legal corpus dataset for training"""
    with open("legal_corpus.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("formatted_dataset.jsonl", "w", encoding="utf-8") as out_file:
        for law in data:
            law_id = law.get("law_id", "unknown")
            for article in law.get("content", []):
                article_text = article.get("content_Article", "").strip()
                if article_text:
                    item = {"text": f"{law_id}: {article_text}"}
                    json.dump(item, out_file, ensure_ascii=False)
                    out_file.write("\n")

def load_training_dataset():
    """Load the formatted dataset for training"""
    dataset = load_dataset("json", data_files="formatted_dataset.jsonl", split="train")
    return dataset

def train_model(model, tokenizer, dataset, max_seq_length):
    """Train the model on the legal corpus"""
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            num_train_epochs=1,
            learning_rate=2e-5,
            embedding_learning_rate=2e-6,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=3407,
            output_dir="outputs",
        ),
    )
    
    trainer_stats = trainer.train()
    return trainer_stats

def push_to_hub(model, tokenizer, hf_token, model_name="thailevann/Qwen3-1.7B_CT_VLSP_track5"):
    """Push the trained model to Hugging Face Hub"""
    login(token=hf_token)
    model.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)

def main():
    """Main training pipeline"""
    print("Starting Qwen3-1.7B continual pretraining...")
    
    
    # Check GPU
    check_gpu()
    
    # Load model
    print("Loading model...")
    model, tokenizer, max_seq_length = load_model()
    
    # Prepare dataset
    print("Preparing dataset...")
    prepare_dataset()
    
    # Load training dataset
    print("Loading training dataset...")
    dataset = load_training_dataset()
    
    # Train model
    print("Starting training...")
    trainer_stats = train_model(model, tokenizer, dataset, max_seq_length)
    
    # Push to hub (set HF_TOKEN environment variable)
    print("Pushing to Hugging Face Hub...")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        push_to_hub(model, tokenizer, hf_token)
    else:
        print("Warning: HF_TOKEN environment variable not set. Skipping model upload.")
    
    print("Training completed successfully!")
    return trainer_stats

if __name__ == "__main__":
    main()