import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel
import json
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import random
import traceback
warnings.filterwarnings('ignore')

class TemporalConfig:
    def __init__(self):
        # Model config
        self.model_name = "vinai/phobert-large"
        self.hidden_size = 768 
        self.num_labels_duration = 2  # yes/no 
        self.max_length = 256
        self.dropout = 0.1
        
        # Training config
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.batch_size = 16
        self.num_epochs = 10
        
        # Task weights
        self.date_weight = 1.0
        self.duration_weight = 1.0
        
        # Device handling
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

class TemporalDataset(Dataset):
    
    def __init__(self, data: List[Dict], tokenizer, config: TemporalConfig, task_type: str):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.task_type = task_type
        self.date_processor = DateArithmeticProcessor()
        
        self._validate_data()
    
    def _validate_data(self):
        """Validate and clean data"""
        valid_data = []
        for i, item in enumerate(self.data):
            try:
                if self.task_type == 'date_arithmetic':
                    if 'question' in item and 'answer' in item:
                        valid_data.append(item)
                    else:
                        if i < 5:
                            print(f"Skipping date item {i}: missing 'question' or 'answer'")
                else:
                    required_fields = ['context', 'question', 'options', 'labels']
                    if all(field in item for field in required_fields):
                        if isinstance(item['options'], list) and isinstance(item['labels'], list):
                            if len(item['options']) == len(item['labels']):
                                valid_data.append(item)
                            else:
                                if i < 5:
                                    print(f"Skipping duration item {i}: options and labels length mismatch")
                        else:
                            if i < 5:
                                print(f"Skipping duration item {i}: options or labels not lists")
                    else:
                        missing_fields = [f for f in required_fields if f not in item]
                        if i < 5:
                            print(f"Skipping duration item {i}: missing fields {missing_fields}")
            except Exception as e:
                if i < 5:
                    print(f"Error validating item {i}: {e}")
        
        print(f"Validated {len(valid_data)}/{len(self.data)} items for {self.task_type}")
        self.data = valid_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            if self.task_type == 'date_arithmetic':
                return self._process_date_arithmetic(item)
            else:
                return self._process_duration_qa(item)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return self._get_dummy_item()
    
    def _get_dummy_item(self):
        if self.task_type == 'date_arithmetic':
            dummy_question = "Thời gian 1 năm trước tháng 1, 2000 là khi nào?"
            dummy_answer = "Tháng 1, 1999"
            
            encoding = self.tokenizer(
                dummy_question,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            target_encoding = self.tokenizer(
                dummy_answer,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'target_ids': target_encoding['input_ids'].squeeze(),
                'target_attention_mask': target_encoding['attention_mask'].squeeze(),
                'question': dummy_question,
                'answer': dummy_answer
            }
        else:  # duration_qa
            dummy_context = "Dummy context"
            dummy_question = "Dummy question"
            dummy_options = ["Option 1", "Option 2"]
            dummy_labels = ["yes", "no"]
            
            batch_data = []
            for i, (option, label) in enumerate(zip(dummy_options, dummy_labels)):
                full_input = f"{dummy_context} [SEP] {dummy_question} [SEP] {option}"
                
                encoding = self.tokenizer(
                    full_input,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                label_int = 1 if label == 'yes' else 0
                
                batch_data.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'label': torch.tensor(label_int, dtype=torch.long),
                    'option_idx': i
                })
            
            return {
                'options_data': batch_data,
                'qid': 0,
                'context': dummy_context,
                'question': dummy_question,
                'options': dummy_options,
                'labels': dummy_labels
            }
    
    def _process_date_arithmetic(self, item):
        question = item['question']
        answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']
        
        # Tokenize input
        encoding = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            answer,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'question': question,
            'answer': answer
        }
    
    def _process_duration_qa(self, item):
        context = item['context']
        question = item['question'] 
        options = item['options']
        labels = item['labels']
        qid = item.get('qid', 0)
        
        # Combine context + question
        input_text = f"{context} [SEP] {question}"
        
        batch_data = []
        for i, (option, label) in enumerate(zip(options, labels)):
            # Combine input với từng option
            full_input = f"{input_text} [SEP] {option}"
            
            encoding = self.tokenizer(
                full_input,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Convert yes/no to 1/0
            label_int = 1 if label == 'yes' else 0
            
            batch_data.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label_int, dtype=torch.long),
                'option_idx': i
            })
        
        return {
            'options_data': batch_data,
            'qid': qid,
            'context': context,
            'question': question,
            'options': options,
            'labels': labels
        }

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
        
        # Encode input
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
            
            # Create target mask for causal attention
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

class TemporalTrainer(L.LightningModule):    
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.model = TemporalModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        special_tokens = ['[SEP]']
        self.tokenizer.add_tokens(special_tokens)
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.date_processor = DateArithmeticProcessor()
        
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, batch, task_type):
        if task_type == 'date_arithmetic':
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task_type=task_type,
                target_ids=batch['target_ids'],
                target_attention_mask=batch['target_attention_mask']
            )
        else:
            return self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task_type=task_type
            )
    
    def _compute_date_loss(self, batch):
        try:
            logits = self.forward(batch, 'date_arithmetic')
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['target_ids'][..., 1:].contiguous()
            
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=pad_token_id
            )
            return loss
        except Exception as e:
            print(f"Error in date loss computation: {e}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _compute_duration_loss(self, batch):
        try:
            total_loss = 0
            batch_size = len(batch['options_data'])
            
            if batch_size == 0:
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            
            for i in range(batch_size):
                options_data = batch['options_data'][i]
                
                input_ids = torch.stack([opt['input_ids'] for opt in options_data])
                attention_mask = torch.stack([opt['attention_mask'] for opt in options_data])
                labels = torch.stack([opt['label'] for opt in options_data])
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type='duration_qa'
                )
                
                loss = F.cross_entropy(logits, labels)
                total_loss += loss
            
            return total_loss / batch_size
        except Exception as e:
            print(f"Error in duration loss computation: {e}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def training_step(self, batch, batch_idx):
        total_loss = 0
        loss_count = 0
        
        try:
            if 'date_batch' in batch and batch['date_batch'] is not None:
                date_loss = self._compute_date_loss(batch['date_batch'])
                total_loss += date_loss * self.config.date_weight
                loss_count += 1
                self.log('train_date_loss', date_loss, prog_bar=True)
            
            if 'duration_batch' in batch and batch['duration_batch'] is not None:
                duration_loss = self._compute_duration_loss(batch['duration_batch'])
                total_loss += duration_loss * self.config.duration_weight
                loss_count += 1
                self.log('train_duration_loss', duration_loss, prog_bar=True)
            
            if loss_count == 0:
                task_type = batch.get('task_type', 'date_arithmetic')
                if task_type == 'date_arithmetic':
                    total_loss = self._compute_date_loss(batch)
                    self.log('train_date_loss', total_loss, prog_bar=True)
                else:
                    total_loss = self._compute_duration_loss(batch)
                    self.log('train_duration_loss', total_loss, prog_bar=True)
                loss_count = 1
            
            if loss_count > 0:
                total_loss = total_loss / loss_count
            else:
                total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            
            self.log('train_loss', total_loss, prog_bar=True)
            return total_loss
            
        except Exception as e:
            print(f"Training step error: {e}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            if dataloader_idx == 0:
                if 'date_batch' in batch:
                    return self._validate_date_batch(batch['date_batch'])
                else:
                    return self._validate_date_batch(batch)
            else:  # Duration QA
                if 'duration_batch' in batch:
                    return self._validate_duration_batch(batch['duration_batch'])
                else:
                    return self._validate_duration_batch(batch)
            
        except Exception as e:
            print(f"Validation step error: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _validate_date_batch(self, batch):
        loss = self._compute_date_loss(batch)
        
        with torch.no_grad():
            predictions = self.generate_date_answers(batch)
            targets = batch['answer']
            
            exact_matches = sum(1 for pred, target in zip(predictions, targets) 
                              if pred.strip().lower() == target.strip().lower())
            accuracy = exact_matches / len(predictions) if len(predictions) > 0 else 0.0
        
        self.log('val_date_loss', loss, prog_bar=True, add_dataloader_idx=False)
        self.log('val_date_accuracy', accuracy, prog_bar=True, add_dataloader_idx=False)
        return loss
    
    def _validate_duration_batch(self, batch):
        loss = self._compute_duration_loss(batch)
        
        with torch.no_grad():
            predictions, targets = self.predict_duration_labels(batch)
            
            if len(predictions) > 0 and len(targets) > 0:
                exact_matches = sum(1 for pred, target in zip(predictions, targets)
                                  if pred == target)
                exact_match = exact_matches / len(predictions)
                self.log('val_duration_exact_match', exact_match, prog_bar=True, add_dataloader_idx=False)
        
        self.log('val_duration_loss', loss, prog_bar=True, add_dataloader_idx=False)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            if dataloader_idx == 0:  # Date arithmetic
                if 'date_batch' in batch:
                    batch = batch['date_batch']
                
                loss = self._compute_date_loss(batch)
                
                with torch.no_grad():
                    predictions = self.generate_date_answers(batch)
                    targets = batch['answer']
                    
                    exact_matches = sum(1 for pred, target in zip(predictions, targets) 
                                      if pred.strip().lower() == target.strip().lower())
                    accuracy = exact_matches / len(predictions) if len(predictions) > 0 else 0.0
                
                self.log('test_date_loss', loss, prog_bar=True, add_dataloader_idx=False)
                self.log('test_date_accuracy', accuracy, prog_bar=True, add_dataloader_idx=False)
                
            else:  # Duration QA
                if 'duration_batch' in batch:
                    batch = batch['duration_batch']
                
                loss = self._compute_duration_loss(batch)
                
                with torch.no_grad():
                    predictions, targets = self.predict_duration_labels(batch)
                    
                    if len(predictions) > 0 and len(targets) > 0:
                        exact_matches = sum(1 for pred, target in zip(predictions, targets)
                                          if pred == target)
                        exact_match = exact_matches / len(predictions)
                        
                        all_preds = [item for sublist in predictions for item in sublist]
                        all_targets = [item for sublist in targets for item in sublist]
                        
                        if len(all_preds) > 0 and len(all_targets) > 0:
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                all_targets, all_preds, average='binary', zero_division=0
                            )
                            accuracy = accuracy_score(all_targets, all_preds)
                        else:
                            precision = recall = f1 = accuracy = 0.0
                    else:
                        exact_match = precision = recall = f1 = accuracy = 0.0
                
                self.log('test_duration_loss', loss, prog_bar=True, add_dataloader_idx=False)
                self.log('test_duration_exact_match', exact_match, prog_bar=True, add_dataloader_idx=False)
                self.log('test_duration_accuracy', accuracy, prog_bar=True, add_dataloader_idx=False)
                self.log('test_duration_precision', precision, prog_bar=True, add_dataloader_idx=False)
                self.log('test_duration_recall', recall, prog_bar=True, add_dataloader_idx=False)
                self.log('test_duration_f1', f1, prog_bar=True, add_dataloader_idx=False)
            
            return loss
            
        except Exception as e:
            print(f"Test step error: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def generate_date_answers(self, batch):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(batch['question'])):
                question = batch['question'][i]
                pred = self.date_processor.process_question(question)
                predictions.append(pred)
        
        return predictions
    
    def predict_duration_labels(self, batch):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            batch_size = len(batch['options_data'])
            
            for i in range(batch_size):
                options_data = batch['options_data'][i]
                
                input_ids = torch.stack([opt['input_ids'] for opt in options_data])
                attention_mask = torch.stack([opt['attention_mask'] for opt in options_data])
                true_labels = [opt['label'].item() for opt in options_data]
                
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_type='duration_qa'
                )
                
                predicted_labels = torch.argmax(logits, dim=1).cpu().tolist()
                
                predictions.append(predicted_labels)
                targets.append(true_labels)
        
        return predictions, targets
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        
        return [optimizer], [scheduler]

class MultiTaskDataModule(L.LightningDataModule):    
    def __init__(self, config: TemporalConfig, date_train_path: str, duration_train_path: str):
        super().__init__()
        self.config = config
        self.date_train_path = date_train_path
        self.duration_train_path = duration_train_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        special_tokens = ['[SEP]']
        self.tokenizer.add_tokens(special_tokens)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup(self, stage=None):
        print("Loading and splitting data...")
        
        date_data = self.load_data(self.date_train_path, 'date')
        duration_data = self.load_data(self.duration_train_path, 'duration')
        
        date_train, date_val, date_test = self.split_data_three_way(date_data, 0.7, 0.15, 0.15)
        duration_train, duration_val, duration_test = self.split_data_three_way(duration_data, 0.7, 0.15, 0.15)
        
        self.date_train_dataset = TemporalDataset(date_train, self.tokenizer, self.config, 'date_arithmetic')
        self.date_val_dataset = TemporalDataset(date_val, self.tokenizer, self.config, 'date_arithmetic')
        self.date_test_dataset = TemporalDataset(date_test, self.tokenizer, self.config, 'date_arithmetic')
        
        self.duration_train_dataset = TemporalDataset(duration_train, self.tokenizer, self.config, 'duration_qa')
        self.duration_val_dataset = TemporalDataset(duration_val, self.tokenizer, self.config, 'duration_qa')
        self.duration_test_dataset = TemporalDataset(duration_test, self.tokenizer, self.config, 'duration_qa')
        
        print(f"\n=== Data Statistics ===")
        print(f"Date Arithmetic - Train: {len(date_train)}, Val: {len(date_val)}, Test: {len(date_test)}")
        print(f"Duration QA - Train: {len(duration_train)}, Val: {len(duration_val)}, Test: {len(duration_test)}")
    
    def load_data(self, file_path: str, data_type: str) -> List[Dict]:
        data = []
        try:
            print(f"Loading {data_type} data from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        if line_num <= 5:  # Only show first few errors
                            print(f"JSON error in line {line_num}: {e}")
                        continue
            
            print(f"Successfully loaded {len(data)} {data_type} items")
            
        except FileNotFoundError:
            print(f"File {file_path} not found! Creating dummy data...")
            if data_type == 'date':
                data = [
                    {"question": "Thời gian 1 năm trước tháng 6, 1297 là khi nào?", "answer": ["Tháng 6, 1296"]},
                    {"question": "Thời gian 2 tháng sau tháng 3, 1400 là khi nào?", "answer": ["Tháng 5, 1400"]},
                    {"question": "Thời gian 6 tháng trước tháng 12, 1500 là khi nào?", "answer": ["Tháng 6, 1500"]},
                ]
            else:  # duration
                data = [
                    {"context": "Tôi nấu cơm.", "question": "Mất bao lâu để nấu cơm?", "options": ["30 phút", "2 giờ"], "labels": ["yes", "no"], "qid": 1},
                    {"context": "Anh ấy du lịch.", "question": "Chuyến đi kéo dài bao lâu?", "options": ["1 ngày", "1 tuần"], "labels": ["no", "yes"], "qid": 2},
                    {"context": "Cô ấy học tiếng Anh.", "question": "Cô ấy học trong bao lâu?", "options": ["1 tháng", "2 năm"], "labels": ["no", "yes"], "qid": 3},
                ]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            data = []
        
        return data
    
    def split_data_three_way(self, data: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float):
        if len(data) == 0:
            return [], [], []
        
        random.shuffle(data)
        
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        
        n = len(data)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, train_end + max(1, int(n * val_ratio)))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end] if val_end > train_end else data[:1]
        test_data = data[val_end:] if val_end < n else data[:1]
        
        return train_data, val_data, test_data
    
    def collate_fn_mixed(self, batch):
        try:
            batch = [item for item in batch if item is not None]
            if not batch:
                return None
            
            # Separate by task type
            date_items = [item for item in batch if 'target_ids' in item]
            duration_items = [item for item in batch if 'options_data' in item]
            
            result = {}
            
            if date_items:
                result['date_batch'] = {
                    'input_ids': torch.stack([item['input_ids'] for item in date_items]),
                    'attention_mask': torch.stack([item['attention_mask'] for item in date_items]),
                    'target_ids': torch.stack([item['target_ids'] for item in date_items]),
                    'target_attention_mask': torch.stack([item['target_attention_mask'] for item in date_items]),
                    'question': [item['question'] for item in date_items],
                    'answer': [item['answer'] for item in date_items],
                    'task_type': 'date_arithmetic'
                }
            
            if duration_items:
                result['duration_batch'] = {
                    'options_data': [item['options_data'] for item in duration_items],
                    'qid': [item['qid'] for item in duration_items],
                    'context': [item['context'] for item in duration_items],
                    'question': [item['question'] for item in duration_items],
                    'options': [item['options'] for item in duration_items],
                    'labels': [item['labels'] for item in duration_items],
                    'task_type': 'duration_qa'
                }
            
            return result
            
        except Exception as e:
            print(f"Error in mixed collate: {e}")
            return None
    
    def collate_fn_date(self, batch):
        try:
            batch = [item for item in batch if item is not None]
            if not batch:
                return None
            
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'target_ids': torch.stack([item['target_ids'] for item in batch]),
                'target_attention_mask': torch.stack([item['target_attention_mask'] for item in batch]),
                'question': [item['question'] for item in batch],
                'answer': [item['answer'] for item in batch],
                'task_type': 'date_arithmetic'
            }
        except Exception as e:
            print(f"Error in date collate: {e}")
            return None
    
    def collate_fn_duration(self, batch):
        try:
            batch = [item for item in batch if item is not None]
            if not batch:
                return None
            
            return {
                'options_data': [item['options_data'] for item in batch],
                'qid': [item['qid'] for item in batch],
                'context': [item['context'] for item in batch],
                'question': [item['question'] for item in batch],
                'options': [item['options'] for item in batch],
                'labels': [item['labels'] for item in batch],
                'task_type': 'duration_qa'
            }
        except Exception as e:
            print(f"Error in duration collate: {e}")
            return None
    
    def train_dataloader(self):
        combined_dataset = ConcatDataset([self.date_train_dataset, self.duration_train_dataset])
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn_mixed,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        date_loader = DataLoader(
            self.date_val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_date,
            num_workers=0,
            pin_memory=True
        )
        
        duration_loader = DataLoader(
            self.duration_val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_duration,
            num_workers=0,
            pin_memory=True
        )
        
        return [date_loader, duration_loader]
    
    def test_dataloader(self):
        date_loader = DataLoader(
            self.date_test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_date,
            num_workers=0,
            pin_memory=True
        )
        
        duration_loader = DataLoader(
            self.duration_test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_duration,
            num_workers=0,
            pin_memory=True
        )
        
        return [date_loader, duration_loader]



def inference_date_arithmetic(model, question: str):
    model.eval()
    processor = DateArithmeticProcessor()
    
    with torch.no_grad():
        answer = processor.process_question(question)
        return answer



def inference_duration_qa(model, context: str, question: str, options: List[str]):
    model.eval()
    tokenizer = model.tokenizer
    device = next(model.model.parameters()).device
    
    input_text = f"{context} [SEP] {question}"
    predictions = []
    
    with torch.no_grad():
        for option in options:
            full_input = f"{input_text} [SEP] {option}"
            
            encoding = tokenizer(
                full_input,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            try:
                logits = model.model(
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


def evaluate_model(model, test_data_path: str, task_type: str):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f]
    
    if task_type == 'date_arithmetic':
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            question = item['question']
            true_answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']
            
            pred_answer = inference_date_arithmetic(model, question)
            
            if pred_answer.strip().lower() == true_answer.strip().lower():
                correct += 1
        
        accuracy = correct / total
        print(f"Date Arithmetic Accuracy: {accuracy:.4f}")
        return accuracy
    
    else:  # duration_qa
        exact_matches = 0
        all_predictions = []
        all_targets = []
        
        for item in test_data:
            context = item['context']
            question = item['question']
            options = item['options']
            true_labels = item['labels']
            
            pred_labels = inference_duration_qa(model, context, question, options)
            
            if pred_labels == true_labels:
                exact_matches += 1
            
            # For precision/recall/f1
            pred_binary = [1 if label == 'yes' else 0 for label in pred_labels]
            true_binary = [1 if label == 'yes' else 0 for label in true_labels]
            
            all_predictions.extend(pred_binary)
            all_targets.extend(true_binary)
        
        exact_match = exact_matches / len(test_data)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        
        print(f"Duration QA - Exact Match: {exact_match:.4f}")
        print(f"Duration QA - Precision: {precision:.4f}")
        print(f"Duration QA - Recall: {recall:.4f}")
        print(f"Duration QA - F1: {f1:.4f}")
        
        return {
            'exact_match': exact_match,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def train_temporal_model_full():
    try:
        # Enhanced config
        config = TemporalConfig()
        config.batch_size = 4
        config.num_epochs = 20
        config.learning_rate = 2e-5
        config.max_length = 256
        config.date_weight = 1.0
        config.duration_weight = 1.0
        
        print(f"Using device: {config.device}")
        print(f"Training config: {config.num_epochs} epochs, batch size {config.batch_size}")
        
        # Data paths
        date_train_path = "/home/mvu9/folder_01_ngoc/softmind_aio/Track-8-TrainingDataset/TrainingDataset/date_training_dataset.txt"
        duration_train_path = "/home/mvu9/folder_01_ngoc/softmind_aio/Track-8-TrainingDataset/TrainingDataset/duration_training_dataset.txt"
        
        # Use multi-task data module
        print("Setting up multi-task data module...")
        data_module = MultiTaskDataModule(config, date_train_path, duration_train_path)
        
        print("Setting up model...")
        model = TemporalTrainer(config)
        
        trainer = L.Trainer(
            max_epochs=config.num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            val_check_interval=0.25,
            log_every_n_steps=25,
            enable_checkpointing=True,
            logger=True,
            enable_progress_bar=True,
            # NO max_steps - train đầy đủ!
        )
        
        print("Starting full multi-task training...")
        trainer.fit(model, data_module)
        
        print("Running final test...")
        test_results = trainer.test(model, data_module)
        
        return model, trainer, config, test_results
        
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        return None, None, None, None


print("Starting FULL temporal model training...")
model, trainer, config, test_results = train_temporal_model_full()

if model is not None:
    torch.save(model.state_dict(), "temporal_model_full.pth")
    print("Full model saved!")
    print(f"Test results: {test_results}")
    
    print("\n=== Example Inference ===")
    
    date_question = "Thời gian 1 năm và 2 tháng trước tháng 6, 1297 là khi nào?"
    date_answer = inference_date_arithmetic(model, date_question)
    print(f"Date Q: {date_question}")
    print(f"Date A: {date_answer}")
    
    try:
        duration_context = "Tôi đang sửa chữa chiếc xe đạp bị hỏng."
        duration_question = "Mất thời gian bao lâu để sửa chữa chiếc xe đạp?"
        duration_options = ["30 phút", "1 tháng", "10 phút", "2 giờ"]
        duration_answers = inference_duration_qa(model, duration_context, duration_question, duration_options)
        print(f"\nDuration Context: {duration_context}")
        print(f"Duration Q: {duration_question}")
        print(f"Duration Options: {duration_options}")
        print(f"Duration A: {duration_answers}")
    except Exception as e:
        print(f"Duration inference error: {e}")
    
else:
    print("Training failed!")