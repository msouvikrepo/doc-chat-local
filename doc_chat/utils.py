from sentence_transformers import SentenceTransformer
import pinecone

import streamlit as st

model = SentenceTransformer('google/flan-t5-base')

pinecone.init(api_key='023454eb-f48c-4a78-ba68-e8f2fac6e897', environment='gcp-starter')
index = pinecone.Index('langchain-chatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    print("Querying Pinecone...")
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string