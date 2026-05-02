import torch
import evaluate
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Assuming your folder structure from previous turns
from src.dataaaa.loader import SummarizationDataset 
import sys
import os
# This allows the script to find the 'src' directory regardless of where you run it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def run_validation(model_path, val_csv_path):
    # 1. Load Metric
    rouge = evaluate.load("rouge")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Saved Model and Tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # 3. Load Validation Data
    # We use a small subset (e.g., 100 rows) for a quick check
    df_val = pd.read_csv(val_csv_path, nrows=100).dropna()
    val_dataset = SummarizationDataset(df_val, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # 4. Generate Summaries
    model.eval()
    predictions = []
    references = []

    print("Generating summaries for validation...")
    for batch in val_loader:
        with torch.no_grad():
            # Generate the summary
            output_tokens = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_length=64,
                num_beams=4 # Helps get better quality summaries
            )
            
            # Decode predictions
            preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            
            # Decode references (labels)
            labels = batch['labels']
            labels[labels == -100] = tokenizer.pad_token_id # Fix the -100 for decoding
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)

    # 5. Compute ROUGE Score
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\n--- Validation Results ---")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Show one example
    print("\n--- Sample Comparison ---")
    print(f"Original: {df_val.iloc[0]['article'][:200]}...")
    print(f"Predicted Summary: {predictions[0]}")
    print(f"Actual Highlight: {references[0]}")

if __name__ == "__main__":
    # Update these paths based on where Colab saved your data/model
    MODEL_DIR = "./saved_model" 
    VAL_DATA = "C:/Users/MAHIR/.cache/kagglehub/datasets/.../validation.csv" # Or Colab path
    run_validation(MODEL_DIR, VAL_DATA)
