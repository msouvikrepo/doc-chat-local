import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings

pinecone.init(api_key='023454eb-f48c-4a78-ba68-e8f2fac6e897', environment='gcp-starter')
embeddings = SentenceTransformerEmbeddings(model_name="google/flan-t5-base")
index = Pinecone.from_existing_index(index_name="langchain-chatbot", embedding=embeddings)


def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

query = "Which documents do I need for an Aadhar card?"
similar_docs = get_similiar_docs(query)
print(similar_docs)