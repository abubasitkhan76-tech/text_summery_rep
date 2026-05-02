import sys
import os
import torch
import kagglehub
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW # Removed redundant AutoModel import

# 1. SETUP PATHS
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import your custom modules
from src.dataaaa.loader import SummarizationDataset 
from src.model.model_bart import get_summarization_model # Using your model script

# 2. DOWNLOAD & LOAD DATA
print("Downloading dataset...")
path = kagglehub.dataset_download("gowrishankarp/newspaper-text-summarization-cnn-dailymail")
train_path = os.path.join(path, "cnn_dailymail", "train.csv") 

# Safety check for path
if not os.path.exists(train_path):
    import glob
    csv_files = glob.glob(os.path.join(path, "**", "train.csv"), recursive=True)
    train_path = csv_files[0] if csv_files else train_path

df = pd.read_csv(train_path, nrows=500)

# 3. SETUP MODEL & TOKENIZER
model_name = "facebook/bart-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Using the function from your bartmodel.py
model = get_summarization_model(model_name, device)

# 4. PREPARE DATASET
train_dataset = SummarizationDataset(df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 5. TRAINING LOOP
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

print(f"Starting training on {device}...")
for epoch in range(1):
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")

# 6. SAVE MODEL
save_path = "./trained_bart_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Training complete! Model saved to {save_path}")
