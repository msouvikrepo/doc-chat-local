from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from typing import Dict
import json
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Talk to your document")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

endpoint_name = "#Endpoint name from Sagemaker console"
model_name = "tiiuae/falcon-7b" #Model that you deployed
tokenizer = AutoTokenizer.from_pretrained(model_name)


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    len_prompt = 0

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        self.len_prompt = len(prompt)
        input_str = json.dumps(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "stop": ["Human:"],
                    "do_sample": False,
                    "repetition_penalty": 1.1,
                },
            }
        )
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        ans = res[0]["generated_text"][self.len_prompt :]
        ans = ans[: ans.rfind("Human")].strip()
        return ans

content_handler = ContentHandler()


@st.cache_resource
def load_chain():

    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name="#region",
        content_handler=content_handler,
    )

    return llm


# this is the object we will work with in the app - it contains the LLM info
chatchain = load_chain()

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(k=3, return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context and if the answer is not contained within the text below, say 'I don't know'"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages(
    [
        system_msg_template,
        MessagesPlaceholder(variable_name="history"),
        human_msg_template,
    ]
)

# TODO : Summarize history before feeding it back into conversation chain
conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=chatchain,
    verbose=True,
)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query  # query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            response = conversation.predict(
                input=f"Context:\n {context} \n\n Query:\n{query}"
            )
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state["responses"]:

        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(
                    st.session_state["requests"][i], is_user=True, key=str(i) + "_user"
                )
