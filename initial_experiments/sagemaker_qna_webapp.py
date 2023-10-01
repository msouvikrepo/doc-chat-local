import streamlit as st

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

#Streamlit display
st.set_page_config(page_title="Q&A for Sagemaker", page_icon=":robot:")
question = st.text_input('Enter Your Query: ')

#LLM specs
model_name = "google/flan-t5-base"
embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@st.cache_resource
def load_chain():
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=local_llm, memory=memory)
    return chain

chatchain = load_chain()

#Document -> Embeddings -> Vector store
loader = TextLoader('assets/Sagemaker_qna.txt')
qna = loader.load()
text_splitter_qna = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts_qna = text_splitter_qna.split_documents(qna)
vectordb = Chroma.from_documents(texts_qna, embedding_function)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":2})

template_qna_format = """Question: {question}
Answer: Give me an answer for the question {question} while making use of the information and knowledge obtained from the document"""

template_qna = PromptTemplate(template=template_qna_format, input_variables=["question"])

#chainQna = LLMChain(llm=local_llm, prompt=template_qna, verbose=True, output_key='answer', memory=memoryQna)
docRetriever = RetrievalQA.from_chain_type(llm=load_chain,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True)

# Display the output if the the user gives an input
if question:

    with st.spinner('Generating response...'):
        response = docRetriever({"query": question}, return_only_outputs=True)
        answer = response["result"]
        st.write(answer)

    with st.expander("Source Documents: "):
        st.info(response["source_documents"])