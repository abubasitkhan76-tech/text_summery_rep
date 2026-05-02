import pandas as pd
import re
from torch.utils.data import DataLoader
from src.dataaaa.loader import SummarizationDataset 

def clean_text(text):
    """
    Removes 'By...PUBLISHED' headers and '(CNN) --' tags that confuse the model.
    """
    if not isinstance(text, str):
        return ""
    # 1. Strip 'By . [Author] . PUBLISHED: ... [Date] .'
    text = re.sub(r'^.*?PUBLISHED:.*?\d{2}:\d{2} [A-Z]{3}, \d+ \w+ \d{4} \. ', '', text, flags=re.S)
    # 2. Strip '(CNN) --'
    text = re.sub(r'^\(CNN\)\s*--\s*', '', text)
    # 3. Fix double spacing and weird dot patterns
    text = text.replace(" . ", " ").replace("..", ".").strip()
    return text

def get_dataloader(file_path, tokenizer, batch_size=4, nrows=None):
    # Load data (using nrows=None will load the whole file on Colab)
    df = pd.read_csv(file_path, nrows=nrows)
    
    # Apply the deep cleaning to the 'article' column
    df['article'] = df['article'].apply(clean_text)
    
    # Use your existing SummarizationDataset from loader.py
    dataset = SummarizationDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
