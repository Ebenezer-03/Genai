"""
Experiment 1: Text Generation using Transformers
Aim: To generate text continuation from a prompt using a Hugging Face pre-trained model.
"""

from transformers import pipeline

def main():
    print("=" * 60)
    print("EXPERIMENT 1: TEXT GENERATION")
    print("=" * 60)

    # Load text generation pipeline using distilgpt2 for speed
    print("\nLoading text-generation pipeline (distilgpt2)...")
    generator = pipeline("text-generation", model="distilgpt2")

    prompt = "Artificial Intelligence is transforming the future of"
    print(f"\nPrompt: {prompt}")

    print("\nGenerating text...")
    results = generator(prompt, max_new_tokens=40, pad_token_id=50256)

    print("\nGenerated Text Output:")
    print("-" * 40)
    print(results[0]["generated_text"])
    print("-" * 40)

if __name__ == "__main__":
    main()
