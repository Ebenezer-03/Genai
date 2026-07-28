"""
Experiment 7: Code Generation and Debugging Assistant
Aim: Use language models to generate code snippets and identify bugs in existing code.
"""

from transformers import pipeline

def run_code_assistant():
    print("=" * 60)
    print("EXPERIMENT 7: CODE GENERATION & DEBUGGING")
    print("=" * 60)

    generator = pipeline("text-generation", model="distilgpt2")

    # 1. Code Generation
    code_prompt = "# Python function to calculate the factorial of a number:\ndef factorial(n):\n"
    print("\n--- 1. Code Generation ---")
    print(f"Prompt:\n{code_prompt}")
    gen_result = generator(code_prompt, max_new_tokens=40, pad_token_id=50256)
    print("Generated Code:")
    print("-" * 40)
    print(gen_result[0]["generated_text"])
    print("-" * 40)

    # 2. Code Debugging
    buggy_code = (
        "# Find and explain the bug in this Python snippet:\n"
        "def add_numbers(a, b):\n"
        "    return a + b\n"
        "print(add_numbers('5', 10))\n"
        "# Error: Cannot concatenate str and int. Solution:"
    )
    print("\n--- 2. Code Debugging Assistant ---")
    print(f"Prompt:\n{buggy_code}")
    debug_result = generator(buggy_code, max_new_tokens=40, pad_token_id=50256)
    print("Debugging Output:")
    print("-" * 40)
    print(debug_result[0]["generated_text"])
    print("-" * 40)

if __name__ == "__main__":
    run_code_assistant()
