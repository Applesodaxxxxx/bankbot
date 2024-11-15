import os
import csv
import json
import pandas as pd
from typing import Text, Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.events import SlotSet

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

# Get best match from knowledge base
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
            dispatcher.utter_message(text="An error occurred. Please try again later.")
            print(f"Error: {e}")

        return [UserUtteranceReverted()]

# Check Balance Action
class ActionCheckBalance(Action):
    def name(self) -> str:
        return "action_check_balance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_id = tracker.get_slot("user_id")
        print(f"The type of user_id is: {type(user_id)}")

        if not user_id:
            dispatcher.utter_message(text="I couldn't retrieve your user ID. Please provide it.")
            return []

        try:
            user_record = user_data[user_data["user_id"] == int(user_id)]
            print(f"The type of user_id is: {type(user_record)}")
            if not user_record.empty:
                balance = user_record["balance"].iloc[0]
                dispatcher.utter_message(response="utter_provide_balance", user_id=user_id, balance=balance)
            else:
                dispatcher.utter_message(text="I couldn't find your account information. Please try again.")
        except Exception as e:
            dispatcher.utter_message(text="An error occurred. Please try again later.")
            print(f"Error: {e}")
        
        return []

class ActionSubmitRating(Action):
    def name(self) -> Text:
        return "action_submit_rating"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        rating = tracker.get_slot("rating")
        
        if not rating or not rating.isdigit() or not (1 <= int(rating) <= 5):
            dispatcher.utter_message(text="Invalid rating. Please provide a number between 1 and 5.")
            return []

        dispatcher.utter_message(text="Thank you for your rating. Now, please provide your feedback.")
        
        return [SlotSet("expecting_feedback", True), SlotSet("rating", rating)]
    
class ActionSubmitFeedback(Action):
    def name(self) -> Text:
        return "action_submit_feedback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        rating = tracker.get_slot("rating")
        feedback_message = tracker.get_slot("feedback_message")

        if not rating:
            dispatcher.utter_message(text="Please provide a rating first.")
            return []

        feedback_message = feedback_message or "No comments provided"
        
        feedback_file_path = "feedback.csv"
        try:
            with open(feedback_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([rating, feedback_message])
            dispatcher.utter_message(response="utter_feedback_thank_you")
        except Exception as e:
            dispatcher.utter_message(text="An error occurred. Please try again later.")
            print(f"Error: {e}")

        return [SlotSet("rating", None), SlotSet("feedback_message", None),SlotSet("expecting_feedback", False) ]
