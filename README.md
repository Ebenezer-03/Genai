# Generative AI (GenAI) Experiments Repository

Welcome to the **Generative AI Experiments Repository**. This repository contains 8 practical Python experiments demonstrating core concepts in Generative AI, Large Language Models (LLMs), Natural Language Processing (NLP), Prompt Engineering, Retrieval-Augmented Generation (RAG), and Diffusion-based Image Generation.

---

## 📁 Repository Structure

```
Genai/
├── README.md                                   # Comprehensive Lab Documentation
├── experiment1_text_generation.py              # Experiment 1: Text Generation
├── experiment2_prompt_engineering.py           # Experiment 2: Prompt Engineering (Zero/Few-Shot, CoT)
├── experiment3_conversational_chatbot.py        # Experiment 3: Conversational Chatbot
├── experiment4_text_summarization_qa.py        # Experiment 4: Text Summarization & Question Answering
├── experiment5_sentiment_document_classification.py # Experiment 5: Sentiment Analysis & Classification
├── experiment6_rag_system.py                   # Experiment 6: Retrieval-Augmented Generation (RAG)
├── experiment7_code_generation_debugging.py     # Experiment 7: Code Generation & Debugging
├── experiment8_image_generation.py              # Experiment 8: Image Generation using Stable Diffusion
└── outputs/                                    # Folder storing saved outputs for Experiments 1-8
    ├── experiment1_output.txt
    ├── experiment2_output.txt
    ├── experiment3_output.txt
    ├── experiment4_output.txt
    ├── experiment5_output.txt
    ├── experiment6_output.txt
    ├── experiment7_output.txt
    └── experiment8_output.txt
```

---

## 🛠️ Prerequisites & Installation

### Requirements
- **Python 3.10+** (Python 3.14 compatible)
- **PyTorch** (`torch`)
- **Hugging Face Transformers** (`transformers`)
- **Diffusers & Accelerate** (`diffusers`, `accelerate`)

### Installation Command
To install all required packages on your system, run:

```powershell
python -m pip install transformers torch diffusers accelerate Pillow
```

---

## 🧪 Experiments Overview

### 1. [Experiment 1: Text Generation](experiment1_text_generation.py)
* **Aim**: Generate text continuation from a prompt using a pre-trained autoregressive language model (`distilgpt2`).
* **Output File**: `outputs/experiment1_output.txt`
* **Run Command**:
  ```powershell
  python experiment1_text_generation.py
  ```

---

### 2. [Experiment 2: Prompt Engineering Techniques](experiment2_prompt_engineering.py)
* **Aim**: Demonstrate key prompting strategies:
  * **Zero-Shot Prompting**: Direct task instruction without examples.
  * **Few-Shot Prompting**: In-context learning using demonstration exemplars.
  * **Chain-of-Thought (CoT) Prompting**: Step-by-step reasoning prompts.
* **Output File**: `outputs/experiment2_output.txt`
* **Run Command**:
  ```powershell
  python experiment2_prompt_engineering.py
  ```

---

### 3. [Experiment 3: Conversational Chatbot](experiment3_conversational_chatbot.py)
* **Aim**: Implement a conversational assistant that maintains dialogue context across multiple turns.
* **Output File**: `outputs/experiment3_output.txt`
* **Run Command**:
  ```powershell
  python experiment3_conversational_chatbot.py
  ```

---

### 4. [Experiment 4: Text Summarization & Question Answering](experiment4_text_summarization_qa.py)
* **Aim**: Perform document summarization (TL;DR generation) and extractive question answering over context documents.
* **Output File**: `outputs/experiment4_output.txt`
* **Run Command**:
  ```powershell
  python experiment4_text_summarization_qa.py
  ```

---

### 5. [Experiment 5: Sentiment & Document Classification](experiment5_sentiment_document_classification.py)
* **Aim**: Classify text sentiment (`POSITIVE` / `NEGATIVE`) with confidence scores using fine-tuned transformer models (`distilbert-base-uncased-finetuned-sst-2-english`).
* **Output File**: `outputs/experiment5_output.txt`
* **Run Command**:
  ```powershell
  python experiment5_sentiment_document_classification.py
  ```

---

### 6. [Experiment 6: Retrieval-Augmented Generation (RAG) System](experiment6_rag_system.py)
* **Aim**: Build a end-to-end RAG architecture combining similarity-based document retrieval with LLM text generation.
* **Output File**: `outputs/experiment6_output.txt`
* **Run Command**:
  ```powershell
  python experiment6_rag_system.py
  ```

---

### 7. [Experiment 7: Code Generation & Debugging Assistant](experiment7_code_generation_debugging.py)
* **Aim**: Utilize language models to generate code functions and assist in identifying/explaining syntax and runtime bugs.
* **Output File**: `outputs/experiment7_output.txt`
* **Run Command**:
  ```powershell
  python experiment7_code_generation_debugging.py
  ```

---

### 8. [Experiment 8: Image Generation Using Diffusion Models](experiment8_image_generation.py)
* **Aim**: Generate high-resolution digital art images from text prompts using Hugging Face `StableDiffusionPipeline` (Stable Diffusion v1.5).
* **Output File**: `outputs/experiment8_output.txt` & generated image `outputs/generated_city.png`
* **Run Command**:
  ```powershell
  python experiment8_image_generation.py
  ```

---

## 🚀 Running All Experiments

You can execute any script individually or run them sequentially from PowerShell:

```powershell
# Run all experiments sequentially
python experiment1_text_generation.py
python experiment2_prompt_engineering.py
python experiment3_conversational_chatbot.py
python experiment4_text_summarization_qa.py
python experiment5_sentiment_document_classification.py
python experiment6_rag_system.py
python experiment7_code_generation_debugging.py
python experiment8_image_generation.py
```

All execution logs and outputs will be automatically stored inside the `outputs/` directory.
