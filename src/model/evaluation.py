import torch
import evaluate
import pandas as pd
import os
import glob
import kagglehub
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

# 1. Custom Dataset Class (Keep it here so the script is self-contained)
class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512, target_len=64):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = str(self.data.iloc[index]['article'])
        summary = str(self.data.iloc[index]['highlights'])
        source = self.tokenizer(article, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer(summary, max_length=self.target_len, padding='max_length', truncation=True, return_tensors='pt')
        labels = target['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': source['input_ids'].flatten(), 'attention_mask': source['attention_mask'].flatten(), 'labels': labels}

def run_evaluation(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rouge = evaluate.load("rouge")

    # Load Model & Tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(device)

    # Load Data
    path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
    val_file = glob.glob(os.path.join(path, "**", "validation.csv"), recursive=True)[0]
    df_val = pd.read_csv(val_file, nrows=100).dropna()
    
    loader = DataLoader(SummarizationDataset(df_val, tokenizer), batch_size=4)

    # Generate
    model.eval()
    preds, refs = [], []
    print("Generating summaries...")
    for batch in loader:
        with torch.no_grad():
            out = model.generate(input_ids=batch['input_ids'].to(device), max_length=64, num_beams=4)
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            labels = batch['labels']
            labels[labels == -100] = tokenizer.pad_token_id
            refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    # Score
    results = rouge.compute(predictions=preds, references=refs)
    print("\n--- RESULTS ---")
    for k, v in results.items():
        print(f"{k.upper()}: {v:.4f}")

if __name__ == "__main__":
    run_evaluation("./trained_bart_model")
