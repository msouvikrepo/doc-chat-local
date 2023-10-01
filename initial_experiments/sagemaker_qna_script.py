#from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

loader = TextLoader('assets/Sagemaker_qna.txt')
qna = loader.load()

text_splitter_qna = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts_qna = text_splitter_qna.split_documents(qna)

#print(texts_qna)

model_name = "google/flan-t5-base"
embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
vectordb = Chroma.from_documents(texts_qna, embedding_function)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)
local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Question: {question}
Answer: """

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question1 = "What is the capital of Italy?"
print(question1)
print(llm_chain.run(question1))

qa = RetrievalQA.from_chain_type(llm=local_llm,
                                 chain_type="stuff",
                                 retriever=vectordb.as_retriever())

question = "Does sagemaker support R and in which applications?"

llm_output = llm_chain.run(question)
qa_output = qa.run(question)

# LLM without the vector DB
print("LLM Output: ", llm_output)

# LLM with the vector DB
print("Vector DB Output: ", qa_output)