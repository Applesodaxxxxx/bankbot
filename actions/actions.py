import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted

# File paths
base_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(base_dir, "knowledge_base.json")
csv_file_path = os.path.join(base_dir, "user_balance.csv")

# Load Knowledge Base
def load_knowledge_base(file_path):
    with open(file_path, "r") as f:
        kb = json.load(f)

    docs = []
    for category, items in kb.items():
        for item in items:
            question = item.get("question", "")
            response = item.get("response", "")
            docs.append({"question": question, "response": response})
    return docs

knowledge_base = load_knowledge_base(json_file_path)

# Load User Data
def load_user_data():
    try:
        return pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading user data: {e}")
        return pd.DataFrame(columns=["user_id", "balance"])

user_data = load_user_data()

# Initialize TF-IDF
def initialize_tfidf(kb):
    corpus = [item["question"] for item in kb]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = initialize_tfidf(knowledge_base)

# Function to retrieve best match from knowledge base
def retrieve_best_match(query):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = cosine_similarities.argmax()
    best_score = cosine_similarities[best_match_idx]

    if best_score > 0.2:  # Threshold for similarity
        return knowledge_base[best_match_idx]["response"]
    else:
        return "Sorry, I couldn't find an answer in our knowledge base."

# Fallback Action
class ActionCustomFallback(Action):
    def name(self) -> str:
        return "action_custom_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_query = tracker.latest_message.get("text", "").strip()
        if not user_query:
            dispatcher.utter_message(text="I'm sorry, I couldn't understand your query.")
            return []

        try:
            # Process query through TF-IDF and cosine similarity
            response = retrieve_best_match(user_query)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your query.")
            print(f"Error: {e}")

        # Optionally revert user utterance to allow retry
        return [UserUtteranceReverted()]

# Check Balance Action
class ActionCheckBalance(Action):
    def name(self) -> str:
        return "action_check_balance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_id = tracker.get_slot("user_id")

        if not user_id:
            dispatcher.utter_message(text="I couldn't retrieve your user ID. Please provide it.")
            return []

        try:
            # Assuming user_data is a pandas DataFrame loaded from user_balance.csv
            user_record = user_data[user_data["user_id"] == int(user_id)]
            if not user_record.empty:
                balance = user_record["balance"].iloc[0]
                dispatcher.utter_message(response="utter_provide_balance", user_id=user_id, balance=balance)
            else:
                dispatcher.utter_message(text="I couldn't find your account information. Please try again.")
        except Exception as e:
            dispatcher.utter_message(text="An error occurred. Please try again later.")
            print(f"Error: {e}")
        
        return []
