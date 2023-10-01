
# Chat securely with your documents using locally hosted LLM

## Tech Stack
1. Python 3.10 or higher
2. Pinecone as a Vector DB
3. Huggingface and Transformers for locally deploying LLM (We are using Google's Flan T5 Base model with 248M parameters - so that my laptop doesn't die)
4. Langchain for creating the Conversation chain wrapper around the LLM
5. Streamlit for the UI

## Running it locally

Create a virtual environment :

	virtualenv .venv

Install requirements :

	pip3 install requirements.txt

Keep your documents under doc_chat/assets

Indexing the documents to create embeddings and store it in Pinecone :

	python3 indexing.py

Chat with the document :

	streamlit run main.py

## Snapshots

