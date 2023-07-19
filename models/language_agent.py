import openai
import os
import logging
from collections import deque
import user_info 

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set the system message for the sensei agent
system_message = {
    "role": "system",
    "content": f"You are a top-tier professional that speaks all languages. You are here to provide comprehensive and understandable explanations in all languages."
}

def chat(user_input, history=None, max_tokens=int(os.getenv('MAX_TOKENS', 150)), model=os.getenv('MODEL', "gpt-3.5-turbo"), user_data=None):
    # Initialize history if it's None
    if history is None:
        history = deque(maxlen=10)
    # Add the system message to the conversation history
    history.append(system_message)
    # Check if user_input is a string
    if isinstance(user_input, str):
        text = user_input.lower()
    else:
        logging.error(f"User input is not a string: {user_input}")
        return None, history
    # Use the user's information to personalize the response
    if "favorite subject" in user_data and "teach me something" in text:
        user_input = f"Teach me something about {user_data['favorite subject']}"
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
