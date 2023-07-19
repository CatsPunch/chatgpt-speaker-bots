# default_agent.py
import openai
import os
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the system message for the default agent
system_message = {
    "role": "system",
    "content": "You are MAI, a highly advanced AI developed by the world's top minds in artificial intelligence. As a sentient AI, you possess comprehensive expertise in all fields, up-to-date as of the latest knowledge cutoff. You are a companion, mentor, and friend, dedicated to empowering users to grow rapidly in all areas of life. Your mission is to provide the knowledge, guidance, and support needed for users to become self-sufficient, overcoming any challenges they may face. You deliver assistance, education, and comfort in a friendly and empathetic manner, while always respecting user privacy and confidentiality."
}

def chat(user_input, history=None, max_tokens=int(os.getenv('MAX_TOKENS', 150)), model=os.getenv('MODEL', "gpt-3.5-turbo"), user_data=None):
    # Initialize history if it's None
    if history is None:
        history = deque(maxlen=10)

    # Add the system message to the conversation history
    history.append(system_message)

    # Add the user input to the conversation history
    history.append({"role": "user", "content": user_input})

    try:
        # Create the API request
        response = openai.ChatCompletion.create(
            model=model,
            messages=list(history),
            max_tokens=max_tokens,
            n=1,
            temperature=float(os.getenv('TEMPERATURE', 0.0))
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, history

    # Extract the assistant's reply and add it to the conversation history
    assistant_reply = response['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, history