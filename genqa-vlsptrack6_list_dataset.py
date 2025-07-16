from transformers import AutoModelForCausalLM, AutoTokenizer

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
    relevant_content = dataset['train'].select([index])['chunk']
    relevant_meta = dataset['train'].select([index])['metadata']
    

    issue_date = relevant_meta[0]['metadata']['IssueDate']
    formatted_date = f"ngày {issue_date.day} tháng {issue_date.month} năm {issue_date.year}"
    relevant = f"""
    Theo luật số {relevant_meta[0]['metadata']['DocIdentity']}, {relevant_meta[0]['metadata']['OrganName']} ban hành luật {relevant_meta[0]['metadata']['DocName']}  vào {formatted_date}:
    {relevant_content}
    """
    return relevant
    
import json
import re

def clean_json_block(text):
    # Loại bỏ ```json và ```
    text = re.sub(r"^```json\n", "", text.strip())
    text = re.sub(r"\n```$", "", text.strip())
    return json.loads(text)

import random
import json

output_path = "output1.jsonl"

for j in range(len(dataset['train'])):
    # 1. Lấy 1 đoạn tài liệu chính
    relevant1 = get_relevant(j)

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
                print(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                continue
        else:
            qa = qa_raw
        # Ghi nếu hợp lệ
        if isinstance(qa, dict):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
   
    
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
        relevant_str += relevant2 + "\n"

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
                print(f"❌ Lỗi parse JSON tại index {j}, task: {task}: {e}")
                continue
        else:
            qa = qa_raw
        # Ghi nếu hợp lệ
        if isinstance(qa, dict):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
   
    print(f"✅ Đã sinh và ghi dữ liệu cho mẫu #{j}")

from huggingface_hub import HfApi
from huggingface_hub import upload_file

repo_id = "thailevann/QA_VLSP_track6"
api = HfApi()

try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True  # Không lỗi nếu repo đã tồn tại
    )
    print(f"✅ Repo `{repo_id}` đã được tạo (hoặc đã tồn tại).")
except HfHubHTTPError as e:
    print(f"❌ Lỗi khi tạo repo: {e}")

# 3. Upload file
try:
    upload_file(
        path_or_fileobj="output.jsonl",
        path_in_repo="data/output.jsonl",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("✅ Đã upload file output.jsonl lên Hugging Face.")
except Exception as e:
    print(f"❌ Upload thất bại: {e}")