import json
from collections import defaultdict

class UserInfo:
    def __init__(self):
        self.interests = defaultdict(int)
        self.user_data = {}

    def collect_user_info(self):
        self.user_data['name'] = input("What's your name?")
        self.user_data['age'] = input("How old are you?")
        self.user_data['gender'] = input("What's your gender?")
        self.user_data['location'] = input("Where do you live?")
        self.user_data['occupation'] = input("What's your occupation?")
        # Add more questions here as needed

    def add_interest(self, interest):
        self.interests[interest] += 1

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.interests, f)

    @classmethod
    def load_from_file(cls, filename):
        try:
            with open(filename, 'r') as f:
                interests = json.load(f)
        except json.JSONDecodeError:
            interests = {}  # Use an empty dictionary if the file is empty
        user_info = cls()
        user_info.interests = defaultdict(int, interests)
        return user_info

    def store_user_info(self, user_data):
        # Load existing data
        try:
            with open('user_info.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = {}
        # Update existing data with new data
        for key, value in user_data.items():
            if key in existing_data:
                if isinstance(value, list):
                    existing_data[key].extend(value)
                elif isinstance(value, dict):
                    existing_data[key].update(value)
                else:
                    existing_data[key] = value
            else:
                existing_data[key] = value
        # Write updated data back to file
        with open('user_info.json', 'w') as f:
            json.dump(existing_data, f, indent=4)  # Use 4 spaces for indentation

    def retrieve_user_info(self):
        try:
            with open('user_info.json', 'r') as f:
                self.user_data = json.load(f)
        except json.JSONDecodeError:
            self.user_data = {}  # Use an empty dictionary if the file is empty

    def update_user_info(self, new_info):
        # Update the user information with the new information
        self.user_data.update(new_info)
        # Save the updated user information
        self.store_user_info(self.user_data)  # Use the store_user_info method to save the user data
