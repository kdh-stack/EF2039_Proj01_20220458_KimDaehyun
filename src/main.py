"""
EF2039 Project 01 - Sentiment Analysis CLI App
Author: 20220458 Kim Daehyun
"""

import argparse
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    def predict_sentiment(text: str) -> dict:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).numpy().flatten()

        label_id = int(np.argmax(probs))
        label = model.config.id2label[label_id]
        confidence = float(probs[label_id])

        return {"text": text, "label": label, "confidence": confidence}

    return predict_sentiment

def run_interactive_mode(predict_fn):
    print("=== Sentiment Analysis Interactive Mode ===")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in ["quit", "exit"]:
            break
        if not text:
            continue
        result = predict_fn(text)
        print(f"Label: {result['label']}, Confidence: {result['confidence']:.3f}\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="*", help="Text(s) to analyze.")
    return parser.parse_args()

def main():
    args = parse_args()
    predict_fn = load_pipeline()

    if args.text:
        for t in args.text:
            result = predict_fn(t)
            print(f"Input: {t}")
            print(f"Label: {result['label']} (conf={result['confidence']:.3f})\n")
    else:
        run_interactive_mode(predict_fn)

if __name__ == "__main__":
    main()
