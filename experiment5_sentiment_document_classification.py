"""
Experiment 5: Sentiment Analysis and Document Classification
Aim: Perform sentiment classification and text categorization using pre-trained NLP models.
"""

from transformers import pipeline

def run_classification():
    print("=" * 60)
    print("EXPERIMENT 5: SENTIMENT & DOCUMENT CLASSIFICATION")
    print("=" * 60)

    # 1. Sentiment Analysis
    print("\n--- 1. Sentiment Analysis ---")
    classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    texts = [
        "This product exceeded all my expectations, absolutely wonderful!",
        "The service was terrible and the staff was extremely rude.",
        "It was an average movie, nothing special but okay."
    ]

    print("Analyzing sentiment for sample sentences...")
    results = classifier(texts)

    for text, res in zip(texts, results):
        print(f"\nText:  '{text}'")
        print(f"Label: {res['label']} (Confidence: {res['score']:.4f})")

if __name__ == "__main__":
    run_classification()
