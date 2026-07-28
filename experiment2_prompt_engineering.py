"""
Experiment 2: Prompt Engineering Techniques
Aim: Demonstrate Zero-Shot, Few-Shot, and Chain-of-Thought prompting for text generation tasks.
"""

from transformers import pipeline

def main():
    print("=" * 60)
    print("EXPERIMENT 2: PROMPT ENGINEERING")
    print("=" * 60)

    generator = pipeline("text-generation", model="distilgpt2")

    # 1. Zero-Shot Prompting
    zero_shot_prompt = "Translate the following English sentence to French: Hello, how are you?"
    print("\n--- 1. Zero-Shot Prompt ---")
    print(f"Prompt:\n{zero_shot_prompt}")
    res1 = generator(zero_shot_prompt, max_new_tokens=20, pad_token_id=50256)
    print(f"Output:\n{res1[0]['generated_text']}\n")

    # 2. Few-Shot Prompting
    few_shot_prompt = (
        "Classify sentiment as Positive or Negative.\n"
        "Input: Great product! -> Sentiment: Positive\n"
        "Input: Terrible experience. -> Sentiment: Negative\n"
        "Input: Loved the customer service! -> Sentiment:"
    )
    print("--- 2. Few-Shot Prompt ---")
    print(f"Prompt:\n{few_shot_prompt}")
    res2 = generator(few_shot_prompt, max_new_tokens=10, pad_token_id=50256)
    print(f"Output:\n{res2[0]['generated_text']}\n")

    # 3. Chain-of-Thought (CoT) Prompting
    cot_prompt = (
        "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\n"
        "A: Let's think step by step.\n"
        "Roger started with 5 balls. 2 cans of 3 tennis balls is 6 tennis balls. 5 + 6 = 11.\n"
        "The answer is 11.\n\n"
        "Q: Sarah has 10 apples. She gives 3 to her friend and eats 2. How many apples does she have left?\n"
        "A: Let's think step by step.\n"
    )
    print("--- 3. Chain-of-Thought Prompt ---")
    print(f"Prompt:\n{cot_prompt}")
    res3 = generator(cot_prompt, max_new_tokens=40, pad_token_id=50256)
    print(f"Output:\n{res3[0]['generated_text']}\n")

if __name__ == "__main__":
    main()
