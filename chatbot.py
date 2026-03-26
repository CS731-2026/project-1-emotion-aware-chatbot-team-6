import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file in current directory

# 1. Setup API Key (from .env or system env)
api_key = os.getenv("OPENROUTER_API_KEY")

# 2. Initialize Conversation History
# The "system" role sets the rules and personality for the AI behind the scenes.
conversation_history = [
    {"role": "system", "content": "You are a helpful, smart, and concise AI assistant."}
]
print("Welcome to your OpenRouter Chatbot! Type 'exit' or 'quit' to stop.\n")

# 3. Create the continuous chat loop
while True:
    # Ask the user for input
    user_question = input("User: ")
    # Check if the user wants to quit the program
    if user_question.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    # Add the user's new question to the conversation history and only save the last 10 messages (10 user + 10 chatbot)
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
    conversation_history.append({"role": "user", "content": user_question})
    print("AI is thinking...")
    
    # Send the ENTIRE conversation history to OpenRouter
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        data=json.dumps({
            "model": "openrouter/free", # Automatically finds an active, free model
            "messages": conversation_history
        })
    )
    
    # Extract the AI's answer and print it
    try:
        ai_answer = response.json()['choices'][0]['message']['content']
        print(f"\nAI: {ai_answer}\n")
        # Add the AI's answer to the history so it remembers it for the next loop!
        conversation_history.append({"role": "assistant", "content": ai_answer})
        
    # If something goes wrong, print the exact error message from OpenRouter
    except KeyError:
        print("\nOops! Something went wrong. Here is the raw error:")
        print(response.json())
        print("\n")
