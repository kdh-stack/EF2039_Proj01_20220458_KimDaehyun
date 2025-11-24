# EF2039 Project 01 – Sentiment Analysis CLI App  
Author: Daehyun Kim (20220458)  
Course: EF2039 – AI Programming  

---

## Overview

This project implements a **command-line application (CLI)** for sentiment analysis using a **pre-trained Transformer model**.  
The application predicts whether an English sentence expresses a **POSITIVE** or **NEGATIVE** sentiment and also outputs a confidence score.

The goal of this project is to gain hands-on experience with the **entire AI development pipeline**, including:

- Idea selection  
- Pipeline design  
- Model selection  
- Environment construction  
- Model analysis  
- Application development  
- Git-based source code management  
- Distribution and documentation  

---

## Features

- Real AI inference using a pre-trained **DistilBERT** model  
- Two usage modes:
  - **Interactive mode** – type input continuously
  - **Batch mode** – analyze multiple sentences at once
- Lightweight and fast (no training required)
- Reproducible environment with `requirements.txt`
- Clean codebase with comments and modular structure

---

## Model Description 

### ✔ Model Name 
`distilbert-base-uncased-finetuned-sst-2-english`

### ✔ What Is DistilBERT?
- A compressed (distilled) version of the original BERT model  
- Much faster and lighter while preserving ~97% of BERT performance  
- Pre-trained and fine-tuned specifically for **binary sentiment classification (SST-2)**

### ✔ Input / Output Structure

| Stage        | Description                                  |
| ------------ | -------------------------------------------- |
| Input        | Raw English sentence (string)                |
| Tokenization | Converts text → token IDs & attention mask   |
| Model Input  | Tensor shape `(batch_size, sequence_length)` |
| Output       | Logits → softmax → probability distribution  |
| Labels       | `POSITIVE` or `NEGATIVE`                     |

Example:
- Text: "This movie was amazing!"
- Output: POSITIVE (confidence: 1.000)

---

## 🛠️ Environment Setup

### 1. Clone the Repository

git clone https://github.com/kdh-stack/EF2039_Proj01_20220458_KimDaehyun.git
cd EF2039_Proj01_20220458_KimDaehyun


### 2. (Optional) Create Virtual Environment

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


### 3. Install Dependencies

pip install -r requirements.txt

Installed packages include:
- transformers
- torch
- numpy

---

## Project Structure

EF2039_Proj01_20220458_KimDaehyun/
│── src/
│    ├── main.py             # CLI application
│    └── model_utils.py      # Helper functions (model load, analysis)
│
├── requirements.txt
├── README.md
└── ...

---

## Pipeline Design

The overall pipeline of the application:
User Input
      ↓
Tokenizer (DistilBERT)
      ↓
Model Inference (logits → softmax)
      ↓
Sentiment Classification
      ↓
CLI Output (label + confidence)

Step-by-step:
- Receive input text from the user
- Tokenize text into model-readable tensors
- Run inference through DistilBERT
- Compute probability via softmax
- Choose final label
- Print result in a clean format

---

## Usage Guide

### ✔ Interactive Mode (recommended)

python3 src/main.py

Sample run:
=== Sentiment Analysis Interactive Mode ===
Type 'quit' to exit.

Enter text: This movie was amazing!
Label: POSITIVE, Confidence: 1.000

Exit with:
quit

### ✔ Batch Mode (multiple sentences)

python3 src/main.py --text "I love this movie" "This was boring"

Sample output:
- Input: I love this movie
- Label: POSITIVE (conf=0.997)

- Input: This was boring
- Label: NEGATIVE (conf=0.982)

---

## Screenshots (Optional)

Interactive mode example:
=== Sentiment Analysis Interactive Mode ===
Type 'quit' to exit.

Enter text: This movie was amazing!
Label: POSITIVE, Confidence: 1.000
(You may insert an actual screenshot image later if desired.)

---

## Distribution
This project has been fully uploaded to GitHub and includes:
- Complete source code
- Full commit history
- Documentation
- Requirements file for reproducible environment
Anyone can clone the repository and run the CLI application easily.

---

## Future Improvements
- Build a web UI using Streamlit or Gradio
- Extend to multi-class emotion classification
- Add Korean sentiment analysis support
- Deploy as a lightweight API server (FastAPI or Flask)

---

## License
This project is created for educational use under the EF2039 course.