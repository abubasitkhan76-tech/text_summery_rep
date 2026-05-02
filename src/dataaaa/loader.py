import torch
import pandas as pd
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512, target_len=64):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 1. Get raw text
        article = str(self.data.iloc[index]['article']) if pd.notnull(self.data.iloc[index]['article']) else ""
        summary = str(self.data.iloc[index]['highlights']) if pd.notnull(self.data.iloc[index]['highlights']) else ""


        # 2. Tokenize Article
        source = self.tokenizer(
            article, 
            max_length=self.max_len, 
            padding='max_length',
            return_tensors='pt', 
            truncation=True
        )

        # 3. Tokenize Summary
        target = self.tokenizer(
            summary, 
            max_length=self.target_len, 
            padding='max_length',
            return_tensors='pt', 
            truncation=True
        )

        labels = target['input_ids'].flatten()
        
        # 4. Handle padding for BART loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source['input_ids'].flatten(),
            'attention_mask': source['attention_mask'].flatten(),
            'labels': labels
        }
