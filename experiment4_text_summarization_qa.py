"""
Experiment 4: Text Summarization and Question Answering
Aim: Perform text summarization and extractive question answering using Hugging Face pipelines.
"""

from transformers import pipeline

def run_summarization_and_qa():
    print("=" * 60)
    print("EXPERIMENT 4: TEXT SUMMARIZATION & QUESTION ANSWERING")
    print("=" * 60)

    context = (
        "Generative Artificial Intelligence (Generative AI) refers to deep-learning models "
        "that can create high-quality text, images, code, and other content based on the data "
        "they were trained on. Unlike traditional AI models that analyze or classify data, "
        "Generative AI focuses on creation. Popular examples include ChatGPT by OpenAI, "
        "Claude by Anthropic, and Stable Diffusion for image generation."
    )

    print("\n--- Context Document ---")
    print(context)

    # 1. Text Summarization
    print("\n--- 1. Text Summarization ---")
    print("Loading summarization pipeline...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    summary = summarizer(context, max_length=50, min_length=20, do_sample=False)
    print("\nSummary Output:")
    print(summary[0]["summary_text"])

    # 2. Extractive Question Answering
    print("\n--- 2. Question Answering ---")
    print("Loading question-answering pipeline...")
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    questions = [
        "What does Generative AI create?",
        "What is an example of an image generation model?"
    ]

    for q in questions:
        result = qa_pipeline(question=q, context=context)
        print(f"\nQuestion: {q}")
        print(f"Answer:   {result['answer']} (Score: {result['score']:.4f})")

if __name__ == "__main__":
    run_summarization_and_qa()
