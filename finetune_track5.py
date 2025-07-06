import os

from huggingface_hub import login

# Load token from environment variable
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN environment variable not set. Model upload will fail.")

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "thailevann/Qwen3-1.7B_CT_VLSP_track5",
    max_seq_length = 8192,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)

# Download datasets using gdown commands from README.md

from datasets import Dataset
import json

# Bước 1: Load legal_corpus.json và tạo map aid -> (law_id, content)
aid2info = {}

with open('legal_corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

for doc in corpus:
    law_id = doc['law_id']
    for article in doc['content']:
        aid = article['aid']
        content = article['content_Article']
        aid2info[aid] = (law_id, content)

# Bước 2: Load train.json và format lại dữ liệu
instruction_output_list = []

with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

for item in train_data:
    question = item['question']
    relevant_laws = item['relevant_laws']

    output_parts = []
    for idx, aid in enumerate(relevant_laws, start=1):
        law_info = aid2info.get(aid)
        if law_info:
            law_id, content = law_info
            output_parts.append(f"Luật liên quan {idx}: {law_id}\n{content}")
        else:
            output_parts.append(f"Luật liên quan {idx}: [Không tìm thấy aid {aid}]")

    instruction_output_list.append({
        "instruction": question,
        "output": "\n\n".join(output_parts)
    })

# Bước 3: Tạo dataset
dataset = Dataset.from_list(instruction_output_list)

print(dataset)

# Lọc những mẫu KHÔNG có lý do (reason_classification rỗng hoặc None)
dataset_without_reasoning = dataset

def convert_conversations_to_chat_format_non_reasoning(examples):
    question = examples.get("instruction", "").strip()
    answer = examples.get("output", "").strip()

    # Bỏ nếu thiếu nội dung
    if not question or not answer:
        return {"conversation": []}

    # Prompt rõ ràng, tự nhiên
    user_prompt = f"""Bạn là một trợ lý AI trong lĩnh vực pháp luật. Vui lòng trích dẫn các điều luật liên quan đến câu hỏi.

    ## Câu hỏi:
    {question}
    """

    chat_conversations = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer}
    ]

    return {"conversation": chat_conversations}

from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset, Dataset

converted_data_non_reasoning = [convert_conversations_to_chat_format_non_reasoning(data) for data in dataset_without_reasoning]
dataset_without_reasoning = Dataset.from_list(converted_data_non_reasoning )
dataset_without_reasoning = standardize_sharegpt(dataset_without_reasoning)

non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset_without_reasoning["conversation"],
    tokenize = False,
)

print(len(non_reasoning_conversations))

import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)

import pandas as pd

data = pd.Series(non_reasoning_subset)

data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        warmup_steps=50,
        learning_rate=2e-5,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./checkpoints",
        save_total_limit=2,
        fp16=True,
    ),
)

trainer_stats = trainer.train()

model.push_to_hub("thailevann/Qwen3-1.7B_SFT_VLSP_track5")
tokenizer.push_to_hub("thailevann/Qwen3-1.7B_SFT_VLSP_track5")