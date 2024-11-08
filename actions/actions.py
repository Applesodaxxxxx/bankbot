import os
import json
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

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

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_file_path = os.path.join(base_dir, "knowledge_base.json")

knowledge_base = load_knowledge_base(json_file_path)

def get_answer(question):
    for doc in knowledge_base:
        if question.lower() in doc["content"].lower():
            return doc["response"] 
    return "Sorry, I couldn't find an answer."

def call_backend_api(user_id, operation):
    if operation == "get_balance":
        return 1234.56
    return None

class ActionAnswerBankingQuestion(Action):
    def name(self) -> str:
        return "action_answer_banking_question"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_question = tracker.latest_message.get('text', '')
        if not user_question:
            dispatcher.utter_message(text="I didn't understand your question.")
            return []

        try:
            answer = get_answer(user_question)
            dispatcher.utter_message(text=answer) 
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request.")
            print(f"Error: {e}") 
        return []

class ActionCheckBalance(Action):
    def name(self) -> str:
        return "action_check_balance"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_id = tracker.get_slot("user_id") 
        balance = call_backend_api(user_id, "get_balance") 
        dispatcher.utter_message(text=f"Your current balance is ${balance}.")
        return []
