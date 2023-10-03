
# Chat securely with your documents using locally hosted LLM / scaled LLM from Sagemaker

## Solution Architecture



## Tech Stack
1. Python 3.10 or higher
2. Pinecone as Vector Store
3. Huggingface and Transformers for locally deploying LLM (We are using Google's Flan T5 Base model with 248M parameters (3GB) - so that my laptop doesn't die)
4. Langchain for Conversation chain wrapper around the LLM
5. Streamlit for the UI
6. AWS Sagemaker for scaled up version

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

Sample user Q&A :

![Screenshot from 2023-10-01 08-32-31](https://github.com/msouvikrepo/doc-chat-local/assets/127878886/76c8ad0a-4e31-4ed9-85f0-a32c689b4ae9)

Fetching query-relevant context from documents :

![Screenshot from 2023-10-01 08-34-31](https://github.com/msouvikrepo/doc-chat-local/assets/127878886/844778a8-d85f-40b6-b07b-845bd332f427)


