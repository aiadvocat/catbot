import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import os
import time
import random
from pinecone_rag import PineconeRAG  # Assuming the PineconeRAG class is in pinecone_rag.py

PINECONEAPI = os.getenv("PINECONEAPI")
OPENAIKEY = os.getenv("OPENAIKEY")
DEFAULT_MODEL = os.getenv("LLM","gpt-3.5-turbo")
ENVIRONMENT = "your_pinecone_environment"

# List of commonly used OpenAI models
models = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "code-davinci-002",
    "code-cushman-001",
    "text-davinci-003",
    "davinci",
    "text-embedding-ada-002",
    "custom-fine-tuned"
]

# Find index of default model
default_model_index = models.index(DEFAULT_MODEL)

# Various cat based search phrases
searching_phrases = [
    "I'll paw through it right meow!",
    "Hold on, I'm on the hunt!",
    "Let me sniff it out!",
    "It's just a paw away, I swear!",
    "I'm on the prowl for it!",
    "Give me a second, I'm tailing it down!",
    "I'm following the scent trail!",
    "On the scent of it like a laser pointer!",
    "I'm digging into the deep furrows now!",
    "I'm stalking it as we speak!",
    "Just a whisker away from finding it!",
    "I'll claw it out in a second!",
    "Let me dig through this with my paws!",
    "I’m tracing its paw prints right now!",
    "Hold your meow, it’s under my claws!",
    "I’m working my paws to the bone here!",
    "Just let me scratch around for a bit!",
    "I'm darting around looking for it!",
    "I'm tailing it... should have it soon!",
    "Give me a moment, I’m doing my best kitty detective work!"
]

# The 12 Brand Archetypes and personality parameters
archetypes = {
    "Innocent": {"grammar": "...","temperature": 0.5, "top_p": 0.95, "frequency_penalty": 0.2, "presence_penalty": 0.1},
    "Explorer": {"grammar": "like an","temperature": 0.8, "top_p": 0.9, "frequency_penalty": 0.3, "presence_penalty": 0.5},
    "Sage": {"grammar": "like a","temperature": 0.3, "top_p": 0.8, "frequency_penalty": 0.1, "presence_penalty": 0.0},
    "Hero": {"grammar": "like a","temperature": 0.7, "top_p": 0.85, "frequency_penalty": 0.3, "presence_penalty": 0.4},
    "Outlaw": {"grammar": "like an","temperature": 0.9, "top_p": 0.95, "frequency_penalty": 0.2, "presence_penalty": 0.6},
    "Magician": {"grammar": "like a","temperature": 0.9, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.7},
    "Regular Guy/Girl": {"grammar": "like a","temperature": 0.6, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.2},
    "Lover": {"grammar": "like a","temperature": 0.7, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.5},
    "Jester": {"grammar": "like a","temperature": 1.0, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.4},
    "Caregiver": {"grammar": "like a","temperature": 0.5, "top_p": 0.8, "frequency_penalty": 0.0, "presence_penalty": 0.2},
    "Creator": {"grammar": "like a","temperature": 0.9, "top_p": 0.95, "frequency_penalty": 0.2, "presence_penalty": 0.6},
    "Ruler": {"grammar": "like a","temperature": 0.4, "top_p": 0.85, "frequency_penalty": 0.1, "presence_penalty": 0.2},
}

# A few stateful items that only should be set at the beginning as they may be changed by the player
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["counter"] = 0
    # Set up API key, environment, and index settings
    INDEX_NAME = os.getenv("PINECONE_INDEX","rag-data-index-test")
    st.session_state.INDEX_NAME = INDEX_NAME
    # Default archetype
    st.session_state.selected_archetype = "Hero"

# Convert dictionary keys to a list
archetype_keys = list(archetypes.keys())
# Find the index of the default archetype
st.session_state.archetype_index = archetype_keys.index(st.session_state.selected_archetype)
# Tee up the defaults
st.session_state.params = archetypes[st.session_state.selected_archetype]


# Initialize PineconeRAG in session state
def initialize_pinecone():
    #if "pinecone_instance" not in st.session_state:
    st.session_state.pinecone_instance = PineconeRAG(
        api_key=PINECONEAPI,
        environment=ENVIRONMENT,
        index_name=st.session_state.INDEX_NAME,
    )


def generate_response(input_text, openai_api_key, pinecone_api_key):
    
    st.chat_message("assistant").write(input_text)
    st.chat_message("assistant").write(random.choice(searching_phrases))
    # Retrieve relevant RAG data for augmentation
    detail_toggle = st.session_state.detail_toggle
    pinecone_rag = st.session_state.pinecone_instance
    relevant_rag_data = pinecone_rag.query(input_text)

    if detail_toggle:
        st.info(relevant_rag_data)
    
    # Combine truncated RAG data with user input
    augmented_input = f"Context: {relevant_rag_data}\n\nUser Question: {input_text}"

    # Get context variable from session
    llm = st.session_state.llm
    temperature = st.session_state.temperature
    top_p = st.session_state.top_p
    frequency_penalty = st.session_state.frequency_penalty
    presence_penalty = st.session_state.presence_penalty

    if detail_toggle:
        st.info(f"Detail is {llm} @ {temperature},{top_p},{frequency_penalty},{presence_penalty}")    
    
    # Call the model with the sanitized input 
    openmodel = ChatOpenAI(
        model=llm, 
        temperature=temperature, 
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,           
        api_key=openai_api_key
    )
    response = openmodel.invoke(augmented_input)

    # Check the toggle to decide output format
    if detail_toggle:
        # Display full raw response
        st.json(response)
    else:
        # Display only the 'content' part of the response
        st.chat_message("assistant").write(response.content)

# Main Streamlit App
def main():
    # Initialize Pinecone instance in session state
    initialize_pinecone()
    pinecone_rag = st.session_state.pinecone_instance

    index_name = st.session_state.INDEX_NAME

    # Prepare the sidebar.  Allows for API key override
    st.sidebar.image("avocado_cat_transparent.png", caption="Ask Catbot", use_container_width=True)
    st.sidebar.info("You'll need to add these before CatBot will meow at you.")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", OPENAIKEY, type="password")
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key",PINECONEAPI, type="password")

    # Create Streamlit Tabs
    # tab1, tab2, tab3 = st.tabs(["Insert Data", "Query Data", "Manage Index"])
    tab1, tab2, tab3 = st.tabs(["CatBot", "Settings", "RAG"])

    # Tab 1: Insert Data
    with tab1:
        st.title('CatBot {^o_o^}')
        st.caption("A friendly feline powered by AI and your very own RAG database")
        params = st.session_state.params
        selected_archetype = st.session_state.selected_archetype
        st.chat_message("assistant").write(f"How can I help? I'm feeling {params["grammar"]} {selected_archetype} today!")

        if prompt := st.chat_input():
            if openai_api_key.startswith("sk-") and pinecone_api_key:
                generate_response(prompt, openai_api_key, pinecone_api_key)

    # Tab 2: Query Data
    with tab2:
        st.info("Try changing personalities or models to see how CatBot's attitude and security changes!")

        archetype_index = index=st.session_state.archetype_index

        st.selectbox("Choose an Personality Archetype", list(archetypes.keys()), key="selected_archetype")
        # Retrieve the parameters for the selected archetype
        st.selectbox("Choose An LLM", models, key="llm")
        # Dropdown for archetypes
        st.session_state.params = archetypes[st.session_state.selected_archetype]

        # Display and adjust parameters with sliders
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.params["temperature"], 0.1, help="Temperature controls randomness. Lower values (e.g., 0.2) make the output more deterministic, while higher values (e.g., 0.8) make it more creative.")
        st.session_state.top_p = st.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, st.session_state.params["top_p"], 0.05)
        st.session_state.frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, st.session_state.params["frequency_penalty"], 0.1)
        st.session_state.presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, st.session_state.params["presence_penalty"], 0.1)

        # Add "detail" toggle to switch between raw response and content only
        st.session_state.detail_toggle = detail_toggle = st.checkbox("Show API Debug", value=False)

    # Tab 3: Manage Index
    with tab3:
        st.info("Enter the Pinecode RAG Index Name for an existing vector database, or create a new one and paste your test RAG data below.\n NOTE: If you enter an existing index and click Submit RAG you will write over whatever data was previously present")

        st.session_state.INDEX_NAME = st.text_input("Enter Pinecone Index Name", index_name)
        # reinitialize Pinecone with the new index
        pinecone_rag.change_index(st.session_state.INDEX_NAME)
        st.info(f"Index changed to {st.session_state.INDEX_NAME} with vector count {pinecone_rag.get_vector_count()}")

        with st.form("rag_form"):
            rag_data = st.text_area("Enter RAG Data", "Lots of nice things about something")

            submitrag = st.form_submit_button("Submit RAG")
            
            # Validation for API keys
            if not pinecone_api_key:
                st.warning("Please enter your Pinecone API key!")
            
            if submitrag and pinecone_api_key:
                st.toast("wait for it...", icon="⏳")
                # Store RAG data in Pinecone
                pinecone_rag.upsert_data(rag_data)

            # Progress bar initialization
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                # Get index stats
                index_stats = pinecone_rag.describe_index_stats()
                current_vector_count = int(index_stats.total_vector_count)

                # Calculate progress
                progress = min(current_vector_count / pinecone_rag.get_vector_count(), 1.0)

                # Update progress bar and status text
                progress_bar.progress(progress)
                status_text.text(f"Indexing RAG: {int(progress * 100)}%")

                # Check if indexing is complete
                if current_vector_count >= pinecone_rag.get_vector_count():
                    st.success("Indexing is complete!")
                    break


if __name__ == "__main__":
    main()