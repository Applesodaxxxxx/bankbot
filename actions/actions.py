import os
import json
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

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
            content = f"Category: {category}\nQ: {question}\nA: {response}"
            docs.append({"content": content, "response": response, "question": question})
    return docs

# Load User Data
def load_user_data():
    try:
        return pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading user data: {e}")
        return pd.DataFrame(columns=["user_id", "balance"])

knowledge_base = load_knowledge_base(json_file_path)
user_data = load_user_data()

# Get answer from Knowledge Base
def get_answer(question):
    for doc in knowledge_base:
        if question.lower() in doc["content"].lower():
            return doc["response"] 
    return "Sorry, I couldn't find an answer."


class ActionAnswerBankingQuestion(Action):
    def name(self) -> str:
        return "action_answer_banking_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_question = tracker.latest_message.get('text', '')
        if not user_question:
            dispatcher.utter_message(text="Sorry, I didn't understand your question.")
            return []

        try:
            answer = get_answer(user_question)
            dispatcher.utter_message(text=answer) 
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request.")
            print(f"Error: {e}") 
        return []

# Check User Balance
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
