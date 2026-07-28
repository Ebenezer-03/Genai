"""
Experiment 6: Retrieval-Augmented Generation (RAG) System
Aim: Build a basic RAG system combining Document Retrieval with LLM Text Generation.
"""

import math
from collections import Counter
from transformers import pipeline

class SimpleRAG:
    def __init__(self, documents):
        self.documents = documents
        self.generator = pipeline("text-generation", model="distilgpt2")

    def _tokenize(self, text):
        return text.lower().split()

    def _cosine_sim(self, text1, text2):
        vec1 = Counter(self._tokenize(text1))
        vec2 = Counter(self._tokenize(text2))
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def retrieve(self, query, top_k=1):
        scores = [(self._cosine_sim(query, doc), doc) for doc in self.documents]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k]]

    def generate_answer(self, query):
        retrieved_docs = self.retrieve(query, top_k=1)
        context = retrieved_docs[0] if retrieved_docs else ""
        
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        result = self.generator(prompt, max_new_tokens=30, pad_token_id=50256)
        
        return context, result[0]["generated_text"]

def main():
    print("=" * 60)
    print("EXPERIMENT 6: RETRIEVAL-AUGMENTED GENERATION (RAG)")
    print("=" * 60)

    knowledge_base = [
        "Python is a high-level, interpreted programming language known for readability.",
        "Retrieval-Augmented Generation (RAG) combines document retrieval with language models.",
        "PyTorch and TensorFlow are popular deep learning frameworks for building neural networks."
    ]

    rag_system = SimpleRAG(knowledge_base)

    query = "What is RAG?"
    print(f"\nUser Query: {query}")

    context, answer = rag_system.generate_answer(query)

    print("\n--- 1. Retrieved Document Context ---")
    print(context)

    print("\n--- 2. RAG Generated Response ---")
    print(answer)

if __name__ == "__main__":
    main()
