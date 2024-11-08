import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

knowledge_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_base.json')

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
    query = data.get("query", "")
    answer = ask_question(documents, tfidf_vectorizer, tfidf_matrix, query)
    return jsonify({"answer": answer})

# Function to ask a question
def ask_question(docs, vectorizer, tfidf_matrix, query):
    best_match_idx, best_score = retrieve_document(vectorizer, tfidf_matrix, query)
    if best_score > 0.2: 
        return docs[best_match_idx]["response"]
    return "I'm sorry, I couldn't find an answer to your question."

# Function to retrieve document
def retrieve_document(vectorizer, tfidf_matrix, query):
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_match_idx]
    return best_match_idx, best_score

if __name__ == '__main__':
    app.run(debug=True, port=5000)
