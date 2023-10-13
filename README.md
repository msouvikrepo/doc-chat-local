
# Chat securely with your documents using private LLM

This project offers a robust and innovative solution for secure communication while safeguarding your sensitive documents. Leveraging the power of Large Language Models (LLMs), this repository empowers users to have confidential conversations and share private files with ease.

## Why secure chat?

In an age where digital privacy is paramount, traditional messaging platforms may not provide the level of security and control that users require. This project was created with the following goals in mind:

- **End-to-End privacy:** Your messages and documents are protected with end-to-end privacy, ensuring that only you and your intended users can access the content.

- **Trustworthy AI:** We harness the capabilities of Large Language Models to facilitate communication and document handling, enhancing the user experience without compromising security.

- **User-Friendly:** The solution is designed to be intuitive and user-friendly, making it accessible to users of all technical backgrounds.

## Features

- **Secure Messaging:** Exchange query messages securely and privately.
- **Private Document Sharing:** Share your sensitive document's knowledge base with others, confident that they remain confidential.
- **Natural Language Processing:** Leverage cutting-edge Language Models for intelligent interactions and document understanding.
- **User Authentication:** Protect your account with robust user authentication to prevent unauthorized access.
- **Open Source:** This project is entirely open source, enabling transparency and collaborative contributions to enhance its security and features.


## Solution Architecture

![doc-chat](https://github.com/msouvikrepo/doc-chat-local/assets/127878886/2db30966-404a-45e9-aaf2-f4014a51eb35)

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


