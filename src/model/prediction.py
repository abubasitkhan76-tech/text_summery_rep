import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_path="./saved_model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

    def summarize(self, text):
        inputs = self.tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"], 
                num_beams=4, 
                max_length=150, 
                early_stopping=True
            )
            
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    summarizer = Summarizer(model_path="facebook/bart-base") # Using base model for now
    test_text = """
    The Apollo 11 mission was the first spaceflight that landed humans on the Moon. 
    Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo 
    Lunar Module Eagle on July 20, 1969. Armstrong became the first person to step 
    onto the lunar surface six hours and 39 minutes later on July 21.
    """
    print(summarizer.summarize(test_text))

