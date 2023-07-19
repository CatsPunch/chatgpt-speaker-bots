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
import user_info 
import json
import spacy
from fuzzywuzzy import fuzz 
from user_info import UserInfo
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')

# Load the user's information
user_info = UserInfo.load_from_file('user_info.json')

# Define the interest categories as a global variable
interest_categories = {
    "Automotive": ["Automotive news and general info", "Car culture", "Convertibles", "Hybrid and electric vehicles", "Luxury", "Minivans", "Motorcycles", "Off-road vehicles", "Performance vehicles", "Sedans", "SUV's", "Trucks", "Vintage cars"],
    "Beauty": ["Body art", "Face care", "General info", "Hair care", "Make-up and cosmetics", "Perfumes and fragrances", "Shaving and grooming", "Skin care", "Spa and medical spa", "Tanning and sun care"],
    "Books and Literature": ["Biographies and memoirs", "Books news and general info", "Business and finance", "Comics", "Cookbooks, food, and wine", "Health, mind, and body", "Mystery and crime", "Nonfiction", "Politics and current events", "Romance"],
    "Business": ["Advertising", "Biotech and biomedical", "Business news and general info", "Business software", "Construction", "Entrepreneurship", "Government", "Green solutions", "Human resources", "Investors and patents", "Leadership", "Marketing", "Nonprofit", "Small business", "Technology"],
    "Careers": ["Career news and general info", "Job fairs", "Job search", "U.S. Military"],
    "Education": ["Adult education", "College life", "Education news and general info", "Graduate school", "Homeschooling", "Language learning", "Online education", "Special education"],
    "Events": ["Entertainment awards", "Holidays", "Movie festivals", "Music festivals and concerts", "Political elections", "Sporting events", "Tech tradeshows"],
    "Family and Parenting": ["Babies and toddlers", "Daycare and preschool", "Elder care", "Parenting K-6 kids", "Parenting teens"],
    "Food and Drink": ["American cuisine", "Barbecues and grilling", "Bars and nightlife", "Beer", "Cajun and Creole", "Chinese cuisine", "Cocktails and beer", "Coffee and tea", "Cooking", "Desserts and baking", "Dining out", "Ethnic foods", "Fast food", "Fine dining", "Foodie news and general info", "French cuisine", "Italian cuisine", "Japanese cuisine", "Liquor and spirits", "Mexican cuisine", "Vegan", "Vegetarian", "Wine"],
    "Gaming": ["Board gaming", "Computer gaming", "Console gaming", "Gaming news and general info", "Mobile gaming", "Online gaming", "Roleplaying games"],
    "Health": ["Health news and general info"],
    "Hobbies and Interests": ["Arts and crafts", "Astrology", "Birdwatching", "Boating", "Cartoons", "Celebrity fan and gossip", "Chess", "Cigars", "Comedy", "Dance", "Design", "Drawing and sketching", "Exercise and fitness", "Freelance writing", "Gambling", "Genealogy", "Guitar", "Jewelry making", "Modeling", "Needlework", "Painting", "Paranormal phenomena", "Performance arts", "Photography", "Sci-fi and fantasy", "Scrapbooking", "Screenwriting", "Shopping", "Stamps and coins"],
    "Home and Garden": ["Appliances", "Entertaining at home", "Gardening", "General info", "Home repair", "Interior decorating", "Landscaping", "Remodeling and construction"],
    "Law, Government, and Politics": ["Commentary", "Conservative", "Government resources", "Legal issues", "Liberal", "Nonpartisan", "Politics"],
    "Life Stages": ["Auto intenders", "College students", "Dads", "Empty nesters", "Moms", "Newlyweds", "Veterans"],
    "Movies and Television": ["Action and adventure", "Animation", "Bollywood", "Business and news", "Comedy", "Documentary", "Drama", "Foreign", "Horror", "Independent", "Movie news and general info", "Music", "Reality TV", "Romance", "Sci-fi and fantasy", "Sports themed"],
    "Music and Radio": ["Alternative", "Blues", "Christian and gospel", "Classical", "Country", "Dance", "DJs", "Electronic", "Hip hop and rap", "Indie spotlight", "Jazz", "Latino", "Metal", "Music news and general info", "Pop", "R&B and soul", "Reggae", "Rock", "Talk Radio", "Venues", "World"],
    "Personal Finance": ["Banking", "Beginning investing", "Credit, debit, and loans", "Financial news", "Financial planning", "Hedge Funds", "Insurance", "Investing", "Mortgage", "Mutual funds", "Options", "Real estate", "Retirement planning", "Stocks", "Tax planning"],
    "Pets": ["Birds", "Cats", "Dogs", "General info", "Horses", "Reptiles"],
    "Science": ["Biology", "Chemistry", "Geography", "Geology", "Physics", "Science news", "Space and astronomy", "Weather"],
    "Society": ["Dating", "Divorce support", "Marriage", "Senior living", "Weddings"],
    "Sports": ["Action sports", "Auto racing", "Baseball", "Bodybuilding", "Boxing", "Canoeing and kayaking", "Climbing", "College basketball", "College football", "Cricket", "Cycling", "Fantasy sports", "Figure skating", "Fishing", "Golf", "Horse racing", "Hunting and shooting", "Ice hockey", "Martial arts", "Mountain biking", "NASCAR racing", "NBA basketball", "NFL football", "Olympics", "Paintball", "Poker", "Power and motorcycles", "Rodeo", "Rugby", "Running and jogging", "Sailing", "Scuba diving", "Skateboarding", "Skiing", "Snowboarding", "Soccer", "Sporting goods", "Sports news", "Surfing and bodyboarding", "Swimming", "Table tennis and ping-pong", "Tennis", "Volleyball", "Waterskiing and wakeboarding"],
    "Style and Fashion": ["Baby apparel", "Dresses and skirts", "Fashion", "Jewelry", "Kids' apparel", "Men's accessories", "Men's bags", "Men's beachwear", "Men's formal wear", "Men's jeans", "Men's outerwear", "Men's pants", "Men's shoes", "Men's tops", "Sunglasses", "Watches", "Woman's tops", "Women's accessories", "Women's bags", "Women's beachwear", "Women's intimates and hosiery", "Women's jeans", "Women's outerwear", "Women's pants", "Women's shoes", "Women's tops"],
    "Technology and Computing": ["Animation", "Antivirus", "Cameras and camcorders", "Cell phones", "Computer certification", "Computer networking", "Computer programming", "Computer reviews", "Data centers", "Databases", "Enterprise software", "Graphic software", "Home entertainment", "Linux", "MacOS", "Mobile", "Network security", "Open source", "PC support", "SEO", "Startups", "Tablets", "Tech news", "Video conferencing", "Web design", "Windows"],
    "Travel": ["Adventure travel", "Africa", "Air travel", "Asia", "Australia and New Zealand", "Bed and breakfasts", "Business travel", "Camping", "Canada", "Caribbean", "Cruises", "Eastern Europe", "Europe", "France", "Greece", "Hawaii", "Honeymoons and getaways", "Hotels", "Italy", "Japan", "Las Vegas", "Luxury travel", "Mexico and Central America", "National parks", "South America", "Theme parks", "Travel news and general info", "Traveling with kids", "United Kingdom"]
    }

def detect_new_info(text):
    # Define the system message
    system_message = {
        "role": "system",
        "content": "I possess unparalleled expertise in comprehending and organizing all aspects of the user's life, including interests, education, profession, relationships, faith, social connections, and familial ties."
    }
    # Define the user message
    user_message = {
        "role": "user",
        "content": text
    }
    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message]
    )
    # Extract the assistant's message from the response
    assistant_message = response["choices"][0]["message"]["content"]
    # Use the assistant's message to update the user's information
    new_info = parse_assistant_message(assistant_message)
    # Define the keys for single item and list item
    single_item_keys = ["name", "age", "gender", "location", "occupation"]
    list_item_keys = list(interest_categories.keys())
    # Write the new information to the user_info.json file
    write_to_json_file(new_info, single_item_keys, list_item_keys)

def update_user_info(assistant_message):
    # Parse the assistant's message to extract the new information
    new_info = parse_assistant_message(assistant_message)

    # Open the user_info.json file
    with open('user_info.json', 'r+') as file:
        # Load the existing data
        data = json.load(file)

        # Update the data with the new information
        for key, value in new_info.items():
            if key in data:
                data[key].append(value)
            else:
                data[key] = [value]

        # Write the updated data back to the file
        file.seek(0)
        json.dump(data, file, indent=4)

def parse_assistant_message(assistant_message):
    # Initialize an empty dictionary to hold the new information
    new_info = {}
    # Create a Doc object for the assistant's message
    doc = nlp(assistant_message)
    # Iterate over the entities in the Doc
    for ent in doc.ents:
        # Iterate over the interest categories
        for category, sub_categories in interest_categories.items():
            # Iterate over the sub-categories
            for sub_category in sub_categories:
                # Tokenize the sub-category
                sub_category_doc = nlp(sub_category)
                # Check if any of the tokens in the entity's text match any of the tokens in the sub-category
                if any(token.text.lower() in [token.text.lower() for token in sub_category_doc] for token in ent):
                    # If there's a match, add the sub-category to the new information
                    if category not in new_info:
                        new_info[category] = []
                    new_info[category].append(sub_category)
    return new_info

def write_to_json_file(data, single_item_keys, list_item_keys, filename='user_info.json'):
    if os.path.isfile(filename):
        # If file exists, read the existing data
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    else:
        # If file does not exist, initialize an empty dictionary
        existing_data = {}
    # Update the existing data with the new data
    for key, value in data.items():
        if key in existing_data:
            # If the key is already in the existing data
            if key in single_item_keys:
                # If the key is a single item key, replace the existing value
                existing_data[key] = value
            elif key in list_item_keys:
                # If the key is a list item key, append the new value to the list of values
                if not isinstance(existing_data[key], list):
                    # If the existing value is not a list, convert it to a list
                    existing_data[key] = [existing_data[key]]
                # Remove duplicates from value list
                value = list(set(value))
                for val in value:
                    if val not in existing_data[key]:  # Check if the value is already in the list
                        existing_data[key].append(val)
        else:
            # If the key is not in the existing data, add the new key and value
            if key in list_item_keys:
                # If the key is a list item key, initialize it as a list
                existing_data[key] = value if isinstance(value, list) else [value]
            else:
                existing_data[key] = value
    # Write the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Define the question keywords
question_keywords = {
    'name': ['name', 'who am i', 'what is my name', 'who am i', 'do you remember me'],
    'age': ['whats my age', 'how old am i', 'what is my age', 'my age'],
    'gender': ['gender', 'am i male or female', 'my gender', 'what sex am i', 'my sex'],
    'location': ['location', 'where do i live', 'my location', 'where do i reside', 'where is home', 'home'],
    'occupation': ['occupation', 'what do i do', 'what is my job', 'what is my occupation', 'how do i make money', 'what do i do for work']
}

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
    "mai_agent": ["default", "mai", "maimai", "original"],
    "teacher_agent": ["teacher", "sensei", "professor", "wise one", "sage"],
    "language_agent": ["language agent", "foreign language agent", "foreign language tutor", "language tutor", "translation tutor", "language translation tutor"],
    "therapist_agent": ["therapist", "I'm sad", "I'm depressed", "I'm unhappy", "suicide", "kill myself", "no point in living", "death", "died"],
    "fitness_agent": ["fitness", "workout", "exercise", "I need to lose weight", "want to lose weight", "lose weight", "slim down", "bulk", "shred"],
    "health_agent": ["health", "wellness", "nutrition", "diet", "wellbeing", "lifestyle", "self-care", "mindfulness", "meditation", "sleep", "stress management", "physical activity", "healthy eating", "weight management", "hydration", "vitamins", "immunity", "relaxation", "balanced diet", "preventive care", "disease prevention", "holistic health", "doctor", "medical", "I need a physical"],
    "legal_agent": ["legal", "lawyer", "personal injury", "divorce", "employment law", "criminal defense", "estate planning", "immigration", "contracts", "intellectual property", "business law", "family law", "bankruptcy", "tax law", "civil litigation", "medical malpractice", "workers' compensation", "landlord-tenant", "social security disability", "insurance claims", "consumer rights"],
    "speaker_agent": ["translate", "auto translate", "let me speak in {language}", "translate this for me", "let me speak in this person's language"],
    "friend_agent": ["hi friend", "be my friend", "act like my friend", "play with me", "let's play", "let's have fun", "hang with me", "hangout with me", "let's play"],
    "dating_agent": ["relationship", "dating coach", "I'm going on a date", "going on a date", "relationship problems", "problems in my relationship", "relationship help", "dating advice", "how to kiss", "girl problems", "boy problems", "my fetish", "my kinks"],
    "test_agent": ["testing", "test", "practice"]
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

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

@lru_cache(maxsize=1000)  # Cache the most recent 1000 results
def generate_response(input):
    # Use the tokenizer to encode the input
    input_tensor = tokenizer.encode(input, return_tensors='pt')
    # Use the model to generate a response
    output = model.generate(input_tensor, max_length=150)
    # Decode the output to get the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def respond_to_user(texts, history, user_data, current_agent):
    responses = []
    for text in texts:
        # Check if the user provided new information about themselves
        new_info = detect_new_info(text)
        if new_info:
            # Update the user_data dictionary and the user_info.json file
            user_data.update(new_info)
            user_info.store_user_info(user_data)
            responses.append(f"I will remember this information for future reference.")
            continue
        # Check if the user asked to switch agents
        lower_text = text.lower()
        new_agent = None
        for agent, keywords in agent_keywords.items():
            if any(re.search(fr"\b{keyword}\b", lower_text) for keyword in keywords):  # Use regular expressions for keyword matching
                new_agent = agent
                break
        if new_agent and new_agent != current_agent:
            current_agent = new_agent
            history = deque([], maxlen=4096)  # Reset the history when switching to a different agent
            responses.append(f"Switched to {current_agent}.")
            continue
        # If the user didn't ask to switch agents, check if they asked for their information
        response = []
        for key in user_data:
            if key in lower_text:
                response.append(f"Your {key} is {user_data[key]}.")
        if response:
            responses.append(" ".join(response))
            continue
        # If the user didn't ask for their information, proceed with the normal chatbot response
        final, history = agents[current_agent].chat(text, history, max_tokens=int(os.getenv('MAX_TOKENS', 150)), model=os.getenv('MODEL', "gpt-3.5-turbo"), user_data=user_data)
        # Generate a response using the cached function
        try:
            text_response = generate_response(text)
            history_response = generate_response(" ".join(history))
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            continue
        # Use the TF-IDF vectorizer to transform the text and history
        text_vector = vectorizer.transform([text_response])
        history_vector = vectorizer.transform([history_response])
        # Calculate the cosine similarity between the text and history vectors
        similarity = cosine_similarity(text_vector, history_vector)
        # If the similarity is above a certain threshold, use the history response
        if similarity > 0.5:
            final = history_response
        responses.append(final)
        # Print the response before adding it to the list
        print(f"Generated response: {final}")
        responses.append(final)

    # Print the responses before converting them to strings
    print(f"Responses before conversion: {responses}")

    # Convert the responses to strings
    responses = [str(response) for response in responses]
    return responses, history, user_data

#Initialize the current agent
current_agent = 'mai_agent'

# Initialize the text-to-speech engine
speaker = pyttsx3.init()
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)  # Select the voice of Sabina
speaker.setProperty('rate', 150)  # Set the speed of the voice

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize user_info
    user_info = UserInfo()
    # Check if user_info.json exists and load data if it does
    if os.path.exists('user_info.json'):
        user_info = UserInfo.load_from_file('user_info.json')
        user_data = user_info.user_data
    else:
        user_info.collect_user_info()
        user_data = user_info.user_data
    # Initialize the chat history with the system message
    history = deque([agents[current_agent].system_message], maxlen=4096)
    while True:
        try:
            # Record audio from the user
            audio_data = mic.record_audio()
            # Transcribe the audio data into text
            text = mic.transcribe_forever(audio_data)
            # Process the user's input and get a response
            responses, history, user_data = respond_to_user([text], history, user_data, current_agent)
            for response in responses:
                # Speak the response
                speaker.say(response)
                speaker.runAndWait()
                # Log the conversation
                logger.info(f"Me: {text}")
                logger.info(f"MAI: {response}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
        finally:
            # Optional: Save the conversation history after each interaction
            with open('history.json', 'w') as f:
                json.dump(list(history), f, indent=4)  # Use 4 spaces for indentation
            # Optional: Save the user's information after each interaction
            user_info.store_user_info(user_data)

if __name__ == "__main__":
    main()
    # Collect user information at the beginning of the conversation
    user_data = user_info.collect_user_info()
    # Store user information
    user_info.store_user_info(user_data)
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
            if text.strip().lower() == 'update my information':
                user_info.update_user_info() # Update user information if the user wants to
            if text.strip().lower() == 'what is my information':
                user_data = user_info.retrieve_user_info() #Retrieve user information when needed
                print(user_data)
                continue
            # Detect new information in the user's input
            new_info = detect_new_info(text)
            # If new information was found, update the user's information
            if new_info:
                user_info.update_user_info(new_info)
                user_data.update(new_info)  # Update the local copy of the user's information
            # Manage the conversation history
            history = manage_history(history)
            # Check if the user's message contains any personal information
            final, history = respond_to_user(text, history, user_data)
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
