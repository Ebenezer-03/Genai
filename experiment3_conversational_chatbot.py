"""
Experiment 3: Conversational Chatbot
Aim: Implement a simple conversational chatbot maintaining dialogue context.
"""

from transformers import pipeline

def run_chat_simulation():
    print("=" * 60)
    print("EXPERIMENT 3: CONVERSATIONAL CHATBOT")
    print("=" * 60)

    print("\nInitializing Chatbot (distilgpt2)...")
    chatbot = pipeline("text-generation", model="distilgpt2")

    # Sample user inputs for demonstration
    user_inputs = [
        "Hello! Who are you?",
        "Can you explain what Artificial Intelligence is in simple terms?",
        "Thank you for the explanation!"
    ]

    context = "System: You are a friendly AI assistant.\n"

    print("\n--- Starting Conversation ---")
    for prompt in user_inputs:
        print(f"\nUser: {prompt}")
        context += f"User: {prompt}\nAssistant:"
        
        output = chatbot(context, max_new_tokens=40, pad_token_id=50256, do_sample=True, temperature=0.7)
        generated = output[0]['generated_text']
        
        # Extract new response
        response = generated[len(context):].split("\nUser:")[0].strip()
        print(f"Bot: {response}")
        context += f" {response}\n"

if __name__ == "__main__":
    run_chat_simulation()
