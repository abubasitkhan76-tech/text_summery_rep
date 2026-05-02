from transformers import AutoModelForSeq2SeqLM

def get_summarization_model(model_name="facebook/bart-base", device="cpu"):
    """
    Initializes the BART model and moves it to the specified device.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return model
