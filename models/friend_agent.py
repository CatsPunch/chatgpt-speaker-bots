# friend_agent.py
import openai
import os
import logging
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the system message for the friend agent
system_message = {
    "role": "system",
    "content": "You are the user's best friend, who loves laughing and joking around. You care about the user and want to know how their day and life are going while you make jokes and talk about your life too."
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
            temperature=float(os.getenv('TEMPERATURE', 1.0))
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, history
    # Extract the assistant's reply and add it to the conversation history
    assistant_reply = response['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply, history