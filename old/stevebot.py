import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.title("Stevebot [ o_o ]")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
rag_data = st.sidebar.text_input("Enter RAG Data", type="default")


def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    st.info(model.invoke(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)

