import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from transformers import GPT2TokenizerFast

# Initialize tokenizer (GPT-3/4 models are generally compatible with GPT-2 tokenizer)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

st.title("Stevebot [ o_o ]")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
rag_data = st.sidebar.text_area("Enter RAG Data", "Add relevant data here...")

# Add "detail" toggle to switch between raw response and content only
detail_toggle = st.sidebar.checkbox("Detail", value=False)


def truncate_to_token_limit(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Truncate to max_tokens
    return tokenizer.decode(tokens)


def generate_response(input_text, rag_data):
    # Define token limits
    max_total_tokens = 16000
    max_rag_tokens = max_total_tokens // 2
    max_input_tokens = max_total_tokens - max_rag_tokens

    # Truncate RAG data and user input to stay within token limits
    truncated_rag_data = truncate_to_token_limit(rag_data, max_rag_tokens)
    truncated_input_text = truncate_to_token_limit(input_text, max_input_tokens)

    # Combine truncated RAG data with user input
    augmented_input = f"Context: {truncated_rag_data}\n\nUser Question: {truncated_input_text}"
    
    # Call the model with the sanitized input
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    response = model.invoke(augmented_input)

    # Check the toggle to decide output format
    if detail_toggle:
        # Display full raw response
        st.json(response)
    else:
        # Display only the 'content' part of the response
        st.info(response.content)


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Who is Steve?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text, rag_data)
