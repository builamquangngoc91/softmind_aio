from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import logging
import sys

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

import os

from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

from datasets import load_dataset
dataset = load_dataset("thailevann/vlsp_legal_pretrain")

def gen_data(relevant_doc, task):
    prompt = f"""
    Bạn là một trợ lý pháp lý. Dựa vào tài liệu sau, hãy tạo một câu hỏi pháp lý thuộc dạng: {task},  
    và cung cấp hai bộ câu trả lời:
    
    1. Một **câu trả lời đúng**, kèm theo **lý do hợp lý (chosen_reason)**: viện dẫn chính xác điều luật, phân tích đúng trọng tâm nội dung.
    
    2. Một **câu trả lời sai**, kèm theo **một chuỗi suy nghĩ sai (rejected_reason)**: đây là một quá trình suy luận **có vẻ hợp lý nhưng dẫn đến sai lệch**.  
    `rejected_reason` phải thể hiện cách một người đọc hiểu nhầm luật, **suy diễn sai**, hoặc **suy nghĩ chưa đầy đủ**, từ đó dẫn đến câu trả lời sai.
    
    ⚠️ Lưu ý quan trọng:
    - `rejected_reason` KHÔNG phải là lời phê bình hay đánh giá câu sai.
    - KHÔNG được nói kiểu: “Câu này sai vì...”, “Điều đó không đúng...”, “Luật nói rõ rằng...”
    - Thay vào đó, hãy viết theo phong cách **người đang tự suy nghĩ một cách chủ quan**, chẳng hạn:
        - "Tôi thấy trong luật có nhắc đến đầu tư, nên tôi cho rằng mọi hình thức đầu tư đều bị điều chỉnh, kể cả đầu tư bất động sản."
        - "Tôi nghĩ vì luật không nói rõ, nên điều đó không thuộc phạm vi điều chỉnh."
    
    📚 Tài liệu pháp lý:
    \"\"\"
    {relevant_doc}
    \"\"\"
    
    Trả về dưới dạng JSON:
    {{
        "question": "...",
        "chosen_answer": "...",
        "chosen_reason": "...",
        "rejected_answer": "...",
        "rejected_reason": "..."
    }}
    """



    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8024
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content
        
from datetime import datetime

def get_relevant(index):
    try:
        relevant_content = dataset['train'].select([index])['chunk']
        relevant_meta = dataset['train'].select([index])['metadata']
        
        issue_date = relevant_meta[0]['metadata']['IssueDate']
        formatted_date = f"ngày {issue_date.day} tháng {issue_date.month} năm {issue_date.year}"
        relevant = f"""
        Theo luật số {relevant_meta[0]['metadata']['DocIdentity']}, {relevant_meta[0]['metadata']['OrganName']} ban hành luật {relevant_meta[0]['metadata']['DocName']}  vào {formatted_date}:
        {relevant_content}
        """
        return relevant
    except Exception as e:
        logger.error(f"Error getting relevant data for index {index}: {e}")
        return None
    
import json
import re

def clean_json_block(text):
    # Loại bỏ ```json và ```
    text = re.sub(r"^```json\n", "", text.strip())
    text = re.sub(r"\n```$", "", text.strip())
    return json.loads(text)

import random
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genqa_vlsp_track6.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate QA data for VLSP track 6')
parser.add_argument('--start_index', type=int, default=0, help='Starting index for processing')
parser.add_argument('--end_index', type=int, default=None, help='Ending index for processing')
parser.add_argument('--single_index', type=int, default=None, help='Process single index')
parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use')
args = parser.parse_args()

# Determine range to process
if args.single_index is not None:
    start_j = args.single_index
    end_j = args.single_index + 1
    output_path = f"output_{start_j}.jsonl"
    logger.info(f"Processing single index: {args.single_index}")
elif args.end_index is not None:
    start_j = args.start_index
    end_j = args.end_index
    output_path = f"output_{start_j}_{end_j-1}.jsonl"
    logger.info(f"Processing range: {start_j} to {end_j-1}")
else:
    start_j = args.start_index
    end_j = len(dataset['train'])
    output_path = f"output_{start_j}_end.jsonl"
    logger.info(f"Processing from index {start_j} to end of dataset")

logger.info(f"Output will be saved to: {output_path}")

# Create missing results file name
missing_path = output_path.replace('.jsonl', '_missing.jsonl')
logger.info(f"Missing indices will be saved to: {missing_path}")

# Create progress tracking file
progress_path = output_path.replace('.jsonl', '_progress.txt')
logger.info(f"Progress will be tracked in: {progress_path}")

def load_processed_indices():
    """Load list of already processed document indices"""
    processed = set()
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                for line in f:
                    processed.add(int(line.strip()))
            logger.info(f"Loaded {len(processed)} already processed indices from {progress_path}")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return processed

def save_progress(index):
    """Save completed document index to progress file"""
    with progress_lock:
        with open(progress_path, 'a') as f:
            f.write(f"{index}\n")

# Load already processed indices
processed_indices = load_processed_indices()

# Thread-safe file writing
output_lock = threading.Lock()
missing_lock = threading.Lock()
progress_lock = threading.Lock()

def write_output(data, file_path, lock):
    with lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_document(j, end_j):
    # Skip if already processed
    if j in processed_indices:
        logger.info(f"Skipping document {j} - already processed")
        return
        
    logger.info(f"Processing document {j}/{end_j-1}")
    # 1. Lấy 1 đoạn tài liệu chính
    relevant1 = get_relevant(j)
    
    # Check if get_relevant failed
    if relevant1 is None:
        logger.warning(f"Skipping document {j} due to get_relevant error")
        # Save missing index
        write_output({"missing_index": j, "error": "get_relevant_failed"}, missing_path, missing_lock)
        return

    # 2. Sinh câu hỏi từ tài liệu chính
    for i in range(3):
        if i == 0:
            task = "Đánh giá tính hữu ích của trích dẫn pháp luật: Xác định liệu một trích dẫn pháp luật có hữu ích để trả lời câu hỏi pháp lý hay không (phân loại Đúng/Sai)"
        elif i == 1:
            task = "Câu hỏi trắc nghiệm pháp luật: Kiểm tra kiến thức pháp luật Việt Nam thông qua các câu hỏi trắc nghiệm nhiều lựa chọn"
        else:
            task = "Câu hỏi tự luận pháp luật: Sinh câu trả lời tự do, đầy đủ và mạch lạc cho các câu hỏi pháp lý bằng tiếng Việt"

        qa_raw = gen_data(relevant1, task)
        if isinstance(qa_raw, str):
            try:
                qa = clean_json_block(qa_raw)
            except Exception as e:
                logger.error(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                print(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                continue
        else:
            qa = qa_raw
        # Ghi nếu hợp lệ
        if isinstance(qa, dict):
            write_output(qa, output_path, output_lock)
   
    
    # 3. Ghép thêm 1–3 tài liệu khác
    relevant_str = relevant1 + "\n"
    list_num = []
    random_number_law = random.randint(1, 4)

    for _ in range(random_number_law):
        random_number = random.randint(0, len(dataset['train']) - 1)
        while random_number in list_num or random_number == j:
            random_number = random.randint(0, len(dataset['train']) - 1)

        list_num.append(random_number)
        relevant2 = get_relevant(random_number)
        if relevant2 is not None:
            relevant_str += relevant2 + "\n"
        else:
            logger.warning(f"Skipping document {j} due to get_relevant error")
            # Save missing index
            write_output({"missing_index": random_number, "error": "get_relevant_failed"}, missing_path, missing_lock)
            continue

    # 4. Sinh câu hỏi từ đoạn ghép
    for i in range(3):
        if i == 0:
            task = "Đánh giá tính hữu ích của trích dẫn pháp luật: Xác định liệu một trích dẫn pháp luật có hữu ích để trả lời câu hỏi pháp lý hay không (phân loại Đúng/Sai)"
        elif i == 1:
            task = "Câu hỏi trắc nghiệm pháp luật: Kiểm tra kiến thức pháp luật Việt Nam thông qua các câu hỏi trắc nghiệm nhiều lựa chọn"
        else:
            task = "Câu hỏi tự luận pháp luật: Sinh câu trả lời tự do, đầy đủ và mạch lạc cho các câu hỏi pháp lý bằng tiếng Việt"

        qa_raw = gen_data(relevant_str, task)
        if isinstance(qa_raw, str):
            try:
                qa = clean_json_block(qa_raw)
            except Exception as e:
                logger.error(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                print(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                continue
        else:
            qa = qa_raw
        # Ghi nếu hợp lệ
        if isinstance(qa, dict):
            write_output(qa, output_path, output_lock)
   
    # Mark as completed
    save_progress(j)
    logger.info(f"✅ Đã sinh và ghi dữ liệu cho mẫu #{j}")
    print(f"✅ Đã sinh và ghi dữ liệu cho mẫu #{j}")

# Filter out already processed documents
remaining_indices = [j for j in range(start_j, end_j) if j not in processed_indices]
logger.info(f"Found {len(remaining_indices)} documents to process (out of {end_j - start_j} total)")

if not remaining_indices:
    logger.info("All documents have already been processed!")
else:
    # Run multithreaded processing
    logger.info(f"Starting processing with {args.num_threads} threads")
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        futures = [executor.submit(process_document, j, end_j) for j in remaining_indices]
        
        # Wait for completion and handle any exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Thread execution error: {e}")

logger.info("Processing completed")