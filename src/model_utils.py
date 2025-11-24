from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_model_and_tokenizer():
    """
    Load a pre-trained DistilBERT model and its tokenizer for sentiment analysis.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def analyze_model(tokenizer, model, sample_text: str) -> Dict[str, Any]:
    """
    Analyze the model input/output structure using a sample text.
    """
    inputs = tokenizer(sample_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    analysis = {
        "model_name": MODEL_NAME,
        "config": model.config.to_dict(),
        "input_ids_shape": tuple(inputs["input_ids"].shape),
        "attention_mask_shape": tuple(inputs["attention_mask"].shape),
        "logits_shape": tuple(outputs.logits.shape),
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
    }
    return analysis
