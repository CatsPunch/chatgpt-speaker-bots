import pyttsx3
import openai
import mic
import os
import re
import logging
import signal
import sys
import pinecone
import time
from dotenv import load_dotenv
from importlib import import_module
from glob import glob
from collections import deque
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Select the voice of Sabina
engine.setProperty('rate', 150)  # Set the speed of the voice

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='asia-northeast1-gcp')
index = pinecone.Index(index_name=os.getenv("PINECONE_INDEX_NAME"))

# Initialize the sentence transformer model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dynamically load the agents from the models directory
agents = {os.path.basename(f)[:-3]: import_module(f"models.{os.path.basename(f)[:-3]}") for f in glob("models/*.py")}

# Define the keywords for each agent
agent_keywords = {
    "mai_agent": ["default", "mai", "maimai", "my", "mymy", "original", "my very best friend", "my favorite life partner"],
    "teacher_agent": ["learn", "teach", "analyze", "practice", "teacher", "sensei", "professor", "wise one", "sage"],
    "therapist_agent": ["therapist", "I'm sad", "I'm depressed", "I'm unhappy", "suicide", "kill myself", "no point in living", "death", "died"],
    "fitness_agent": ["fitness", "workout", "exercise", "I need to lose weight", "want to lose weight", "lose weight", "slim down", "bulk", "shred"],
    "health_agent": ["health", "wellness", "nutrition", "diet", "wellbeing", "lifestyle", "self-care", "mindfulness", "meditation", "sleep", "stress management", "physical activity", "healthy eating", "weight management", "hydration", "vitamins", "immunity", "relaxation", "balanced diet", "preventive care", "disease prevention", "holistic health", "doctor", "medical", "I need a physical"],
    "legal_agent": ["legal", "lawyer", "personal injury", "divorce", "employment law", "criminal defense", "estate planning", "immigration", "contracts", "intellectual property", "business law", "family law", "bankruptcy", "tax law", "civil litigation", "medical malpractice", "workers' compensation", "landlord-tenant", "social security disability", "insurance claims", "consumer rights"],
    "speaker_agent": ["translate", "auto translate", "let me speak in {language}", "translate this for me", "let me speak in this person's language"],
    "friend_agent": ["friend", "hi friend", "be my friend", "act like my friend", "play with me", "let's play", "let's have fun", "hang with me", "hangout with me", "let's play"],
    "dating_agent": ["dating", "relationship", "dating coach", "I'm going on a date", "going on a date", "relationship problems", "problems in my relationship", "relationship help", "dating advice", "how to kiss", "girl problems", "boy problems", "my fetish", "my kinks"]
}
def speak(text):
    engine.say(text)
    engine.runAndWait()

def manage_history(history, max_tokens=4096):
    """
    This function takes the conversation history and the maximum token limit as input.
    It returns a truncated version of the history if it exceeds the maximum token limit.
    """
    total_tokens = sum([len(c['content'].split()) + len(c['content']) for c in history])
    while total_tokens > max_tokens:
        # Remove the oldest message from the history
        removed_message = history.popleft()
        total_tokens -= len(removed_message['content'].split()) + len(removed_message['content'])
    return history

# Function to handle the KeyboardInterrupt signal
def handle_interrupt(signal, frame):
    if engine.isBusy():
        engine.stop()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def vectorize_conversation(conversation):
    # Convert the conversation history to a single string
    conversation_string = ' '.join([message['content'] for message in conversation])
    # Use the sentence transformer model to vectorize the conversation string
    conversation_vector = model.encode([conversation_string])[0].tolist()
    return conversation_vector

def devectorize_conversation(vector):
    # This function is not implemented because it's not possible to directly convert a vector back into text
    # You might need to store the original text along with the vector in your database
    pass

if __name__ == "__main__":
    # Start with the default agent
    current_agent = "mai_agent"
    history = deque([agents[current_agent].system_message], maxlen=4096)
    while True:
        try:
            audio = mic.record_audio()  # Add a timeout to prevent the function from running indefinitely
            text = mic.transcribe_forever(audio)
            print("You said:" + text)
            if text.strip().upper() == 'STOP':
                if engine.isBusy():
                    engine.stop()
                continue
            if text.strip().lower() == 'SKIP':
                continue
            new_agent = None
            for agent, keywords in agent_keywords.items():
                if any(re.search(fr"\b{keyword}\b", text.lower()) for keyword in keywords):  # Use regular expressions for keyword matching
                    new_agent = agent
                    break
            if new_agent and new_agent != current_agent:
                current_agent = new_agent
                history = deque([agents[current_agent].system_message], maxlen=4096)  # Reset the history only when switching to a different agent
            # Manage the conversation history
            history = manage_history(history)
            final, history = agents[current_agent].chat(text, history)
            # Vectorize the conversation history
            conversation_vector = vectorize_conversation(history)
            # Upsert the conversation history into Pinecone
            upsert_response = index.upsert(
                vectors=[
                    {
                    'id': f'conversation_{time.time()}',  # Use a unique ID for each conversation turn
                    'values': conversation_vector
                    }
                ]
            )
            # Query the conversation history from Pinecone
            query_response = index.query(queries=[conversation_vector], top_k=5)
            # Devectorize the conversation history
            conversation_history = [devectorize_conversation(vector) for vector in query_response.results]
            print(final)
            speak(final)  # Add this line to make the chatbot speak the response
        except KeyboardInterrupt:
            continue
        except Exception as e:
            logging.error(f"An error occurred: {e}")
