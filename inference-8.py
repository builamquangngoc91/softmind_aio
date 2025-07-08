# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
import json
from typing import Dict, List, Optional

class TemporalConfig:
    def __init__(self):
        self.model_name = "vinai/phobert-large"
        self.hidden_size = 768
        self.num_labels_duration = 2
        self.max_length = 256
        self.dropout = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DateArithmeticProcessor:    
    def __init__(self):
        self.month_mapping = {
            'tháng 1': 1, 'tháng 2': 2, 'tháng 3': 3, 'tháng 4': 4,
            'tháng 5': 5, 'tháng 6': 6, 'tháng 7': 7, 'tháng 8': 8,
            'tháng 9': 9, 'tháng 10': 10, 'tháng 11': 11, 'tháng 12': 12
        }
        
        self.time_units = {
            'năm': 'years', 'tháng': 'months', 'ngày': 'days',
            'tuần': 'weeks', 'giờ': 'hours', 'phút': 'minutes'
        }
    
    def parse_vietnamese_date(self, date_str: str) -> datetime:
        date_str = date_str.lower().strip()
        
        # "tháng X, YYYY"
        pattern = r'tháng\s+(\d+),?\s*(\d{4})'
        match = re.search(pattern, date_str)
        if match:
            month, year = int(match.group(1)), int(match.group(2))
            return datetime(year, month, 1)
        
        raise ValueError(f"Cannot parse date: {date_str}")
    
    def parse_time_expression(self, text: str) -> Dict:
        result = {}
        
        # Pattern cho số + đơn vị thời gian
        patterns = {
            'years': r'(\d+)\s*năm',
            'months': r'(\d+)\s*tháng', 
            'days': r'(\d+)\s*ngày',
            'weeks': r'(\d+)\s*tuần',
            'hours': r'(\d+)\s*giờ',
            'minutes': r'(\d+)\s*phút'
        }
        
        for unit, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                result[unit] = int(match.group(1))
        
        return result
    
    def calculate_date(self, base_date: datetime, time_delta: Dict, operation: str) -> datetime:
        delta_kwargs = {k: v for k, v in time_delta.items() if k in ['years', 'months', 'days', 'weeks', 'hours', 'minutes']}
        
        if operation == 'subtract':
            for key in delta_kwargs:
                delta_kwargs[key] = -delta_kwargs[key]
        
        if 'years' in delta_kwargs or 'months' in delta_kwargs:
            years = delta_kwargs.pop('years', 0)
            months = delta_kwargs.pop('months', 0)
            new_date = base_date + relativedelta(years=years, months=months)
            
            if delta_kwargs:
                td = timedelta(**delta_kwargs)
                new_date += td
        else:
            td = timedelta(**delta_kwargs)
            new_date = base_date + td
        
        return new_date
    
    def format_vietnamese_date(self, date: datetime) -> str:
        return f"Tháng {date.month}, {date.year}"
    
    def process_question(self, question: str) -> str:
        try:
            question = question.lower()
            
            if 'trước' in question:
                operation = 'subtract'
            elif 'sau' in question:
                operation = 'add'
            else:
                operation = 'add'  # default
            
            date_match = re.search(r'tháng\s+(\d+),?\s*(\d{4})', question)
            if date_match:
                base_date = self.parse_vietnamese_date(date_match.group(0))
            else:
                raise ValueError("Cannot find base date")
            
            time_delta = self.parse_time_expression(question)
            
            result_date = self.calculate_date(base_date, time_delta, operation)
            
            return self.format_vietnamese_date(result_date)
            
        except Exception as e:
            print(f"Error processing question: {question}, Error: {e}")
            return "Tháng 1, 2000"  # fallback

class TemporalModel(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        actual_hidden_size = self.encoder.config.hidden_size
        print(f"Loaded model hidden size: {actual_hidden_size}")
        
        if actual_hidden_size != config.hidden_size:
            print(f"Updating hidden_size from {config.hidden_size} to {actual_hidden_size}")
            config.hidden_size = actual_hidden_size
        
        # Task-specific heads
        self.date_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        self.date_output = nn.Linear(config.hidden_size, self.encoder.config.vocab_size)
        
        self.duration_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_labels_duration)
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids, attention_mask, task_type, target_ids=None, target_attention_mask=None):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        if target_ids is not None:
            target_ids = target_ids.to(device)
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(device)
        
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded = encoder_outputs.last_hidden_state
        pooled = encoder_outputs.pooler_output
        
        if task_type == 'date_arithmetic':
            return self._forward_date_arithmetic(encoded, target_ids, target_attention_mask)
        else:
            return self._forward_duration_qa(pooled)
    
    def _forward_date_arithmetic(self, encoded, target_ids=None, target_attention_mask=None):
        if target_ids is not None:
            target_embedded = self.encoder.embeddings(target_ids)
            memory = encoded
            tgt = target_embedded
            
            tgt_len = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
            
            decoded = self.date_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits = self.date_output(decoded)
            return logits
        else:
            pooled = encoded.mean(dim=1)
            logits = self.date_output(pooled).unsqueeze(1)
            return logits
    
    def _forward_duration_qa(self, pooled):
        pooled = self.dropout(pooled)
        logits = self.duration_classifier(pooled)
        return logits

def load_model(model_path: str = "temporal_model_full.pth"):
    """Load the trained temporal model"""
    config = TemporalConfig()
    
    # Initialize model
    model = TemporalModel(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens = ['[SEP]']
    tokenizer.add_tokens(special_tokens)
    model.encoder.resize_token_embeddings(len(tokenizer))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    model.to(config.device)
    model.eval()
    
    return model, tokenizer, config

def inference_date_arithmetic(model, tokenizer, config, question: str) -> str:
    """Perform date arithmetic inference"""
    processor = DateArithmeticProcessor()
    
    with torch.no_grad():
        answer = processor.process_question(question)
        return answer

def inference_duration_qa(model, tokenizer, config, context: str, question: str, options: List[str]) -> List[str]:
    """Perform duration QA inference"""
    device = next(model.parameters()).device
    
    input_text = f"{context} [SEP] {question}"
    predictions = []
    
    with torch.no_grad():
        for option in options:
            full_input = f"{input_text} [SEP] {option}"
            
            encoding = tokenizer(
                full_input,
                padding='max_length',
                truncation=True,
                max_length=config.max_length,
                return_tensors='pt'
            )
            
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            try:
                logits = model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    task_type='duration_qa'
                )
                
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
                predictions.append("yes" if prob > 0.5 else "no")
                
            except Exception as e:
                print(f"Error in inference: {e}")
                predictions.append("no")
    
    return predictions

def run_examples():
    """Run example inferences"""
    print("Loading model...")
    model, tokenizer, config = load_model()
    
    print("\n=== Date Arithmetic Examples ===")
    
    # Example 1: Date arithmetic
    date_questions = [
        "Thời gian 1 năm và 2 tháng trước tháng 6, 1297 là khi nào?",
        "Thời gian 3 tháng sau tháng 12, 2020 là khi nào?",
        "Thời gian 2 năm trước tháng 1, 2000 là khi nào?"
    ]
    
    for question in date_questions:
        answer = inference_date_arithmetic(model, tokenizer, config, question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()
    
    print("\n=== Duration QA Examples ===")
    
    # Example 2: Duration QA
    duration_examples = [
        {
            "context": "Tôi đang sửa chữa chiếc xe đạp bị hỏng.",
            "question": "Mất thời gian bao lâu để sửa chữa chiếc xe đạp?",
            "options": ["30 phút", "1 tháng", "10 phút", "2 giờ"]
        },
        {
            "context": "Cô ấy đang nấu cơm cho gia đình.",
            "question": "Cần bao lâu để nấu cơm?",
            "options": ["5 phút", "30 phút", "2 giờ", "1 ngày"]
        },
        {
            "context": "Anh ta đi du lịch châu Âu.",
            "question": "Chuyến du lịch kéo dài bao lâu?",
            "options": ["1 ngày", "2 tuần", "3 tháng", "1 năm"]
        }
    ]
    
    for example in duration_examples:
        predictions = inference_duration_qa(
            model, tokenizer, config,
            example["context"], 
            example["question"], 
            example["options"]
        )
        print(f"Context: {example['context']}")
        print(f"Question: {example['question']}")
        print(f"Options: {example['options']}")
        print(f"Predictions: {predictions}")
        print()

if __name__ == "__main__":
    run_examples()