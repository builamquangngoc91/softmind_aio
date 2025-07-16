import os
import json
import re
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from datetime import datetime, timedelta
import calendar
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer


max_seq_length = 2048
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.75,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


reasoning_start = "<start_reasoning>"
reasoning_end = "<end_reasoning>"
answer_start = "<ANSWER>"
answer_end = "</ANSWER>"

system_prompt = f"""Bạn là một hệ thống AI chuyên xử lý câu hỏi về thời gian.
Khi nhận được câu hỏi, hãy:
1. Suy luận logic về thời gian trong {reasoning_start} và {reasoning_end}
2. Đưa ra câu trả lời chính xác trong {answer_start} và {answer_end}
Với Duration QA, trả lời theo format: option1:yes/no|option2:yes/no|..."""

chat_template = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{{ eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '\n\nHuman: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant: ' + message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n\nAssistant: <start_reasoning>' }}{% endif %}"""

tokenizer.chat_template = chat_template



def load_temporal_data():
    data_list = []

    date_path = "/content/date_training_dataset.txt"
    with open(date_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data_list.append({
                    'task_type': 'date_arithmetic',
                    'question': item['question'],
                    'answer': item['answer'][0] if isinstance(item['answer'], list) else item['answer'],
                    'context': item.get('context', ''),
                })
            except:
                continue

    duration_path = "/content/duration_training_dataset.txt"
    with open(duration_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                answer_pairs = [f"{opt}:{label}" for opt, label in zip(item['options'], item['labels'])]
                answer = "|".join(answer_pairs)

                data_list.append({
                    'task_type': 'duration_qa',
                    'question': item['question'],
                    'answer': answer,
                    'context': item['context'],
                    'options': item['options'],
                    'labels': item['labels']
                })
            except:
                continue

    return pd.DataFrame(data_list)


VIETNAMESE_MONTHS = {
    'tháng 1': 1, 'tháng 2': 2, 'tháng 3': 3, 'tháng 4': 4,
    'tháng 5': 5, 'tháng 6': 6, 'tháng 7': 7, 'tháng 8': 8,
    'tháng 9': 9, 'tháng 10': 10, 'tháng 11': 11, 'tháng 12': 12,
    'tháng một': 1, 'tháng hai': 2, 'tháng ba': 3, 'tháng tư': 4,
    'tháng năm': 5, 'tháng sáu': 6, 'tháng bảy': 7, 'tháng tám': 8,
    'tháng chín': 9, 'tháng mười': 10, 'tháng mười một': 11, 'tháng mười hai': 12
}


def extract_vietnamese_date(text):
    text = text.lower().strip()

    for month_name, month_num in VIETNAMESE_MONTHS.items():
        pattern = rf"{month_name},?\s*(\d{{4}})"
        match = re.search(pattern, text)
        if match:
            year = int(match.group(1))
            return month_num, year

    return None


def parse_duration(text):
    text = text.lower()

    years_match = re.search(r'(\d+)\s*năm', text)
    years = int(years_match.group(1)) if years_match else 0

    months_match = re.search(r'(\d+)\s*tháng', text)
    months = int(months_match.group(1)) if months_match else 0

    days_match = re.search(r'(\d+)\s*ngày', text)
    days = int(days_match.group(1)) if days_match else 0

    hours_match = re.search(r'(\d+)\s*giờ', text)
    hours = int(hours_match.group(1)) if hours_match else 0

    minutes_match = re.search(r'(\d+)\s*phút', text)
    minutes = int(minutes_match.group(1)) if minutes_match else 0

    return {
        'years': years,
        'months': months,
        'days': days,
        'hours': hours,
        'minutes': minutes
    }


def format_temporal_dataset(row):
    task_type = row['task_type']
    question = row['question']
    answer = row['answer']
    context = row.get('context', '')

    if task_type == 'date_arithmetic':
        reasoning = f"Để giải quyết câu hỏi về tính toán ngày tháng này, tôi cần:\n"
        reasoning += "1. Xác định thời điểm ban đầu\n"
        reasoning += "2. Xác định khoảng thời gian cần tính\n"
        reasoning += "3. Thực hiện phép tính cộng/trừ thời gian\n"
        reasoning += "4. Đưa ra kết quả theo định dạng yêu cầu"
    else:
        reasoning = f"Với context: '{context}'\n"
        reasoning += "Tôi cần đánh giá từng lựa chọn thời gian dựa trên:\n"
        reasoning += "1. Thông tin trong context\n"
        reasoning += "2. Kiến thức thực tế về thời lượng của hoạt động\n"
        reasoning += "3. Tính hợp lý của từng lựa chọn"

    full_response = f"{reasoning_start}{reasoning}{reasoning_end}{answer_start}{answer}{answer_end}"

    if context:
        full_question = f"Context: {context}\n\nCâu hỏi: {question}"
    else:
        full_question = question

    return {
        'messages': [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_question},
            {"role": "assistant", "content": full_response}
        ],
        'text': None
    }


def check_format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]

        if reasoning_start in response and reasoning_end in response:
            score += 1.0
        if answer_start in response and answer_end in response:
            score += 2.0

        try:
            if (response.index(reasoning_start) < response.index(reasoning_end) <
                response.index(answer_start) < response.index(answer_end)):
                score += 2.0
        except:
            score -= 1.0

        scores.append(score)
    return scores


def check_date_arithmetic_reward(prompts, completions, answer, **kwargs):
    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        true_answer = answer[i]

        answer_match = re.search(f"{answer_start}(.+?){answer_end}", response, re.DOTALL)
        if not answer_match:
            scores.append(-3.0)
            continue

        predicted = answer_match.group(1).strip()

        if predicted.lower() == true_answer.lower():
            score += 5.0
        else:
            pred_date = extract_vietnamese_date(predicted)
            true_date = extract_vietnamese_date(true_answer)

            if pred_date and true_date:
                if pred_date[0] == true_date[0]:
                    score += 1.5
                if pred_date[1] == true_date[1]:
                    score += 1.5
                if abs(pred_date[1] - true_date[1]) <= 1:
                    score += 0.5
            else:
                score -= 2.0

        scores.append(score)
    return scores


def check_duration_qa_reward(prompts, completions, answer, **kwargs):
    scores = []

    for i, completion in enumerate(completions):
        score = 0
        response = completion[0]["content"]
        true_answer = answer[i]

        answer_match = re.search(f"{answer_start}(.+?){answer_end}", response, re.DOTALL)
        if not answer_match:
            scores.append(-3.0)
            continue

        predicted = answer_match.group(1).strip()

        true_pairs = {}
        for pair in true_answer.split('|'):
            if ':' in pair:
                opt, label = pair.split(':', 1)
                true_pairs[opt.strip()] = label.strip()

        pred_pairs = {}
        for pair in predicted.split('|'):
            if ':' in pair:
                opt, label = pair.split(':', 1)
                pred_pairs[opt.strip()] = label.strip()

        correct = 0
        total = len(true_pairs)

        for opt, true_label in true_pairs.items():
            if opt in pred_pairs and pred_pairs[opt] == true_label:
                correct += 1
                score += 1.0
            else:
                score -= 0.5

        if correct == total:
            score += 3.0

        scores.append(score)
    return scores


def reasoning_quality_reward(completions, **kwargs):
    scores = []

    for completion in completions:
        score = 0
        response = completion[0]["content"]

        reasoning_match = re.search(f"{reasoning_start}(.+?){reasoning_end}", response, re.DOTALL)
        if not reasoning_match:
            scores.append(-1.0)
            continue

        reasoning = reasoning_match.group(1)

        if len(reasoning) > 50:
            score += 1.0
        if len(reasoning) > 100:
            score += 0.5

        temporal_keywords = ['thời gian', 'năm', 'tháng', 'ngày', 'giờ', 'phút',
                           'trước', 'sau', 'khoảng', 'thời điểm', 'thời lượng']
        keyword_count = sum(1 for kw in temporal_keywords if kw in reasoning.lower())
        score += min(keyword_count * 0.3, 2.0)

        if any(marker in reasoning for marker in ['1.', '2.', '3.', 'Bước']):
            score += 1.0

        scores.append(score)
    return scores


print("Loading temporal QA data...")
df_data = load_temporal_data()
print(f"Loaded {len(df_data)} samples")
print(f"Date Arithmetic: {len(df_data[df_data['task_type'] == 'date_arithmetic'])}")
print(f"Duration QA: {len(df_data[df_data['task_type'] == 'duration_qa'])}")

formatted_data = []
for idx, row in df_data.iterrows():
    formatted = format_temporal_dataset(row)
    messages = formatted['messages']

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        n_tokens = len(tokenizer.encode(text))

        formatted_data.append({
            'messages': messages,
            'text': text,
            'n_tokens': n_tokens,
            'answer': row['answer'],
            'task_type': row['task_type']
        })
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

df_formatted = pd.DataFrame(formatted_data)

max_prompt_length = 800
df_filtered = df_formatted[df_formatted['n_tokens'] <= max_prompt_length].copy()
print(f"Filtered to {len(df_filtered)} samples within token limit")

df_filtered['prompt'] = df_filtered['messages'].apply(
    lambda msgs: [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msgs[1]["content"]}
    ]
)

dataset = Dataset.from_pandas(df_filtered[['prompt', 'answer', 'task_type']])


vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=0.95,
    top_k=50,
    temperature=0.8,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
    max_tokens=512,
)


training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=0.8,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=3,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    num_train_epochs=2,
    save_steps=50,
    save_total_limit=3,
    report_to="none",
    output_dir="/content/temporal_qa_model",
    bf16=False,
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
)


def get_task_specific_rewards(batch):
    task_types = batch.get('task_type', ['unknown'] * len(batch['prompt']))

    reward_funcs = [check_format_reward, reasoning_quality_reward]

    if any(t == 'date_arithmetic' for t in task_types):
        reward_funcs.append(check_date_arithmetic_reward)
    if any(t == 'duration_qa' for t in task_types):
        reward_funcs.append(check_duration_qa_reward)

    return reward_funcs

class TemporalQATrainer(GRPOTrainer):
    def compute_rewards(self, prompts, completions, answer, **kwargs):
        batch_idx = kwargs.get('batch_idx', 0)
        start_idx = batch_idx * self.args.per_device_train_batch_size
        end_idx = start_idx + len(prompts)

        task_types = self.train_dataset['task_type'][start_idx:end_idx]

        all_rewards = []

        all_rewards.append(check_format_reward(completions, **kwargs))
        all_rewards.append(reasoning_quality_reward(completions, **kwargs))

        date_indices = [i for i, t in enumerate(task_types) if t == 'date_arithmetic']
        duration_indices = [i for i, t in enumerate(task_types) if t == 'duration_qa']

        if date_indices:
            date_rewards = check_date_arithmetic_reward(
                [prompts[i] for i in date_indices],
                [completions[i] for i in date_indices],
                [answer[i] for i in date_indices],
                **kwargs
            )
            task_rewards = [0.0] * len(prompts)
            for i, idx in enumerate(date_indices):
                task_rewards[idx] = date_rewards[i]
            all_rewards.append(task_rewards)

        if duration_indices:
            duration_rewards = check_duration_qa_reward(
                [prompts[i] for i in duration_indices],
                [completions[i] for i in duration_indices],
                [answer[i] for i in duration_indices],
                **kwargs
            )
            task_rewards = [0.0] * len(prompts)
            for i, idx in enumerate(duration_indices):
                task_rewards[idx] = duration_rewards[i]
            all_rewards.append(task_rewards)

        total_rewards = np.array(all_rewards).sum(axis=0).tolist()
        return total_rewards


trainer = TemporalQATrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        check_format_reward,
        check_date_arithmetic_reward,
        check_duration_qa_reward,
        reasoning_quality_reward,
    ],
    args=training_args,
    train_dataset=dataset,
)


from transformers import TrainerCallback

class TemporalQACallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"\nStep {state.global_step}:")
            for key, value in logs.items():
                if "reward" in key or "loss" in key:
                    print(f"  {key}: {value:.4f}")

trainer.add_callback(TemporalQACallback())

# Train model
print("\nStarting training...")
trainer.train()


# def evaluate_model():
#     print("\n=== Model Evaluation ===")

#     date_test = {
#         "role": "user",
#         "content": "Giả sử bạn đang ở tháng 6, 2020, thời gian sau 2 năm 3 tháng là khi nào?"
#     }

#     duration_test = {
#         "role": "user",
#         "content": "Context: Tôi đang nấu một bữa cơm gia đình.\n\nCâu hỏi: Mất bao lâu để nấu bữa cơm?\nCác lựa chọn: 30 phút, 1 giờ, 5 giờ, 2 ngày"
#     }

#     test_cases = [date_test, duration_test]

#     for test in test_cases:
#         messages = [{"role": "system", "content": system_prompt}, test]
#         text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

#         output = model.fast_generate(
#             text,
#             sampling_params=SamplingParams(temperature=0.1, max_tokens=512),
#             lora_request=None,
#         )[0].outputs[0].text

#         print(f"\nQuestion: {test['content']}")
#         print(f"Answer: {output}")

# evaluate_model()

# print("\nSaving model...")
# model.save_lora("/content/temporal_qa_lora")

# if False:
#     model.save_pretrained_merged(
#         "/content/temporal_qa_merged",
#         tokenizer,
#         save_method="merged_16bit"
#     )

# print("\nTraining completed successfully!")


# def create_inference_pipeline():

#     def predict_temporal_qa(question, context=""):
#         if context:
#             full_question = f"Context: {context}\n\nCâu hỏi: {question}"
#         else:
#             full_question = question

#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": full_question}
#         ]

#         text = tokenizer.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=False
#         )

#         output = model.fast_generate(
#             text,
#             sampling_params=SamplingParams(
#                 temperature=0.1,
#                 top_p=0.9,
#                 max_tokens=512
#             ),
#             lora_request=model.load_lora("/content/temporal_qa_lora"),
#         )[0].outputs[0].text

#         answer_match = re.search(f"{answer_start}(.+?){answer_end}", output, re.DOTALL)
#         if answer_match:
#             return answer_match.group(1).strip()
#         return output

#     return predict_temporal_qa

# predict = create_inference_pipeline()



# def generate_submission(test_file_path, output_path, task_type):
#     predictions = []

#     with open(test_file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 item = json.loads(line.strip())

#                 if task_type == 'date_arithmetic':
#                     pred = predict(item['question'])
#                     predictions.append({
#                         'question': item['question'],
#                         'prediction': pred
#                     })
#                 else:
#                     pred = predict(item['question'], item['context'])
#                     labels = []
#                     for opt in item['options']:
#                         if f"{opt}:yes" in pred:
#                             labels.append("yes")
#                         else:
#                             labels.append("no")

#                     predictions.append({
#                         'qid': item['qid'],
#                         'predictions': labels
#                     })
#             except Exception as e:
#                 print(f"Error processing: {e}")
#                 continue

#     with open(output_path, 'w', encoding='utf-8') as f:
#         for pred in predictions:
#             f.write(json.dumps(pred, ensure_ascii=False) + '\n')

#     print(f"Saved {len(predictions)} predictions to {output_path}")