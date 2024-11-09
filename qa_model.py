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

knowledge_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actions', 'knowledge_base.json')
conversation_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conversation_log.csv')
feedback_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback.csv')

if not os.path.exists(conversation_log_path):
    with open(conversation_log_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['User Query', 'Bot Response'])

if not os.path.exists(feedback_file_path):
    with open(feedback_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Rating', 'Feedback'])

# Load the Knowledge Base
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

documents = load_knowledge_base(knowledge_base_path)

# Initialize the TF-IDF model
def initialize_tfidf(docs):
    corpus = [doc["content"] for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = initialize_tfidf(documents)

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

        # If Rasa responds with text, use it
        if rasa_data and "text" in rasa_data[0]:
            response = rasa_data[0]["text"]
            log_conversation(query, response)
            return jsonify({"answer": response})
    except Exception as e:
        print(f"Error communicating with Rasa server: {e}")

    # Fall back to the knowledge base
    print("Falling back to knowledge base...")
    answer = ask_question(documents, tfidf_vectorizer, tfidf_matrix, query)
    log_conversation(query, answer)
    return jsonify({"answer": answer})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """Save feedback from the user."""
    data = request.json
    rating = data.get('rating', '').strip()
    message = data.get('message', 'No comments provided').strip() 

    # Validate rating
    if not rating or not rating.isdigit() or not (1 <= int(rating) <= 5):
        return jsonify({"error": "Invalid rating. Please provide a number between 1 and 5."}), 400

    # Save feedback to CSV
    with open(feedback_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([rating, message])

    return jsonify({"response": "Thank you for your feedback!"}), 200

def ask_question(docs, vectorizer, tfidf_matrix, query):
    best_match_idx, best_score = retrieve_document(vectorizer, tfidf_matrix, query)
    if best_score > 0.2: 
        return docs[best_match_idx]["response"]
    return "I'm sorry, I couldn't find an answer to your question."

def retrieve_document(vectorizer, tfidf_matrix, query):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_match_idx]
    return best_match_idx, best_score

def log_conversation(user_query, bot_response):
    with open(conversation_log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_query, bot_response])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
