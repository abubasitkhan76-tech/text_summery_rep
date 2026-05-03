from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import uvicorn
import torch

app = FastAPI(title="BART Summarizer API")

# 1. Setup Model Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (BASE_DIR / "trained_bart_model").resolve()

# 2. Define Request Schema
class SummaryRequest(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

# Global variables for the model and tokenizer
model = None
tokenizer = None

# 3. Lifespan: Load model on startup
@app.on_event("startup")
def load_model():
    global model, tokenizer
    print(f"--- 🚀 Loading model from: {MODEL_PATH} ---")
    
    if not (MODEL_PATH / "config.json").exists():
        print(f"--- ❌ ERROR: config.json not found in {MODEL_PATH} ---")
        return

    try:
        # Explicitly loading BART classes
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH))
        print("--- ✅ SUCCESS: Model and Tokenizer Loaded! ---")
    except Exception as e:
        print(f"--- ❌ LOADING ERROR: {e} ---")

# 4. API Routes
@app.get("/")
def home():
    return {"status": "Online", "model_loaded": model is not None}

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    
    try:
        # Tokenize input text
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=1024
        )
        
        # Generate summary IDs
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=request.max_length, 
            min_length=request.min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        # Decode IDs back to text
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {"summary": summary_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
