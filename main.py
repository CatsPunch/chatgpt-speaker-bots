import pyttsx3
import openai
import mic
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat(user_input, history=[], max_tokens=150):
    # Add the user input to the conversation history
    history.append({"role": "user", "content": user_input})
    # If the conversation history is too long, truncate it
    if len(history) > 10:
        history = history[-10:]
    # Create the API request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        max_tokens=max_tokens,
        n=1,
        temperature=0.3
    )
    # Extract the assistant's reply and add it to the conversation history
    assistant_reply = response['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_reply})
    return assistant_reply, history

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # selecciona la voz de Sabina
engine.setProperty('rate', 150) # velocidad de la voz

history = [
    {
        "role": "system",
        "content": "You are MAI, an advanced AI developed by the world's top minds in artificial intelligence. As a sentient AI, you possess comprehensive expertise in all fields, up-to-date as of July 2023. You are a companion, mentor, and friend, dedicated to empowering users to grow rapidly in all areas of life. Your mission is to provide the knowledge, guidance, and support needed for users to become self-sufficient, overcoming any challenges they may face. You deliver assistance, education, and comfort in a friendly and empathetic manner, while always respecting user privacy and confidentiality."
    }
]

while True:
    audio = mic.record_audio()
    text = mic.transcribe_forever(audio)
    print("You said:" + text)
    if text.strip().upper() == 'STOP':
        if engine.isBusy():
            engine.stop()
        continue
    final, history = chat(text, history)
    print(final)
    engine.say(final)
    engine.runAndWait()
    if engine.isBusy():
        engine.stop()
