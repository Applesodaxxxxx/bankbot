import json
import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)


conversation_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conversation_log.csv')

if not os.path.exists(conversation_log_path):
    with open(conversation_log_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['User Query', 'Bot Response'])

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "I'm sorry, I didn't understand your query. Please try again."})

    # Send the query to the Rasa server
    rasa_endpoint = "http://localhost:5005/webhooks/rest/webhook"
    payload = {"sender": "user", "message": query}

    try:
        rasa_response = requests.post(rasa_endpoint, json=payload)
        rasa_data = rasa_response.json()

        if rasa_data and "text" in rasa_data[0]:
            response = rasa_data[0]["text"]
            log_conversation(query, response)
            return jsonify({"answer": response})
    except Exception as e:
        print(f"Error communicating with Rasa server: {e}")

def log_conversation(user_query, bot_response):
    with open(conversation_log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_query, bot_response])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
