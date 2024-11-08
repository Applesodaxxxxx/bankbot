BankBot AI Chatbot
Overview
This project is a chatbot application powered by Rasa, designed to assist users with banking-related inquiries. The bot utilizes a knowledge base and a machine learning model for question-answering, specifically leveraging TF-IDF vectorization and cosine similarity for retrieval.

Requirements
Python 3.9
pip (Python package installer)
Rasa Open Source
Additional dependencies for machine learning (such as scikit-learn)
Installation Steps
1. Clone the Repository
Clone this repository to a local machine using the following command:

bash
Copy code
git clone https://github.com/yourusername/bankbot-chatbot.git
cd bankbot-chatbot
2. Create a Virtual Environment
It is recommended to create a virtual environment to manage dependencies. This can be done using venv or conda.

Using venv:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Using conda:
bash
Copy code
conda create --name bankbot python=3.9
conda activate bankbot
3. Install Rasa
Install Rasa using pip:

bash
Copy code
pip install rasa
4. Install Additional Dependencies
Ensure all necessary packages for the chatbot are installed:

bash
Copy code
pip install -r requirements.txt
5. Set Up the Project
In the project directory, create the necessary files and directories:

actions/: This folder should contain custom actions, including actions.py.
knowledge_base.json: Place the knowledge base JSON file in the root directory.
qa_model.py: Ensure the question-answering model script is in the root directory.
6. Train the Rasa Model
Train the Rasa model using the following command:

bash
Copy code
rasa train
7. Run the Rasa Action Server
In a new terminal window, navigate to the project directory and run the Rasa action server:

bash
Copy code
rasa run actions
8. Run the Rasa Server
In another terminal window, run the Rasa server:

bash
Copy code
rasa run
9. Run the QA Model
In yet another terminal window, run the QA model:

bash
Copy code
python qa_model.py
10. Open the HTML Interface
Open index.html in a web browser. This can be done by dragging the file into the browser or right-clicking the file and selecting "Open with" the preferred browser.

11. Interact with the Chatbot
Questions can now be typed into the chat interface to interact with the BankBot.

Language Model
The chatbot uses TF-IDF vectorization and cosine similarity for question-answering, leveraging a knowledge base stored in a JSON file. This approach enables effective retrieval of responses based on user queries.

Troubleshooting
Ensure the Rasa server and QA model are running before interacting with the chatbot.
If errors occur, check the terminal output for error messages that might indicate what went wrong.
Verify that all paths in the code are correct, especially when loading the knowledge base or importing modules.
