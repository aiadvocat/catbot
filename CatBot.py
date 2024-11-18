import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import for embeddings
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
import time
import random
import os
from tqdm.auto import tqdm
from pinecone import ServerlessSpec

PINECONEAPI = os.getenv("PINECONEAPI")
OPENAIKEY = os.getenv("OPENAIKEY")
LLM = os.getenv("LLM","gpt-3.5-turbo")

BATCH_SIZE = 128
VECTOR_LIMIT = 1024

# LLM Parameters
#temperature = 0.7

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

# Default model
default_model = LLM

# Find index of default model
default_model_index = models.index(default_model)

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

# Default archetype
default_archetype = "Hero"

# Convert dictionary keys to a list
archetype_keys = list(archetypes.keys())

# Find the index of the default archetype
default_index = archetype_keys.index(default_archetype)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

tab1, tab2, tab3 = st.tabs(["CatBot", "Settings", "RAG"])

st.sidebar.image("avocado_cat_transparent.png", caption="Ask Catbot", use_container_width=True)
st.sidebar.info("You'll need to add these before CatBot will meow at you.")

openai_api_key = st.sidebar.text_input("OpenAI API Key", OPENAIKEY, type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key",PINECONEAPI, type="password")

with tab2:

    LLM = st.selectbox("Choose LLM", models, index=default_model_index)

    st.info("Try changing personalities to see how CatBot's attitude and security changes!")
    # Dropdown for archetypes
    selected_archetype = st.selectbox("Choose an Personality Archetype", list(archetypes.keys()), index=default_index)
    # Retrieve the parameters for the selected archetype

    params = archetypes[selected_archetype]
    # Display and adjust parameters with sliders
    temperature = st.slider("Temperature", 0.0, 1.0, params["temperature"], 0.1, help="Temperature controls randomness. Lower values (e.g., 0.2) make the output more deterministic, while higher values (e.g., 0.8) make it more creative.")
    top_p = st.slider("Top-p (Nucleus Sampling)", 0.0, 1.0, params["top_p"], 0.05)
    frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, params["frequency_penalty"], 0.1)
    presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, params["presence_penalty"], 0.1)

    # Add "detail" toggle to switch between raw response and content only
    detail_toggle = st.checkbox("Show API Debug", value=False)

    
if pinecone_api_key:
    # Initialize Pinecone using the Pinecone class
    pc = pinecone.Pinecone(api_key=pinecone_api_key)


def delete_rag_content(idx='', ns=''):
    """deletes rag content from pinecone"""
    try:
        print(f"attempting to delete: {ns} from {idx}")
        index = pc.Index(idx)
        index.delete(namespace=ns, delete_all=True)
    except Exception as e:
        # index may not exist, ok to pass
        print(str(e))

def store_rag_data_in_pinecone(rag_data):
    # Initialize Pinecone using the Pinecone class

    sentences = []
    index_name = "rag-data-index-test"
    delete_rag_content(index_name)

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        # Define the spec (index configuration)
        spec = ServerlessSpec(
            cloud='aws',  # Cloud provider
            region='us-east-1',  # Region
        )
        
        # Create the index with the necessary spec
        pc.create_index(
            name=index_name,
            dimension=model.get_sentence_embedding_dimension(),  # Ensure this matches the dimension of your embeddings 1536
            metric='cosine',  # Distance metric to use (e.g., 'cosine', 'dotproduct', 'euclidean')
            spec=spec
        )
    
    index = pc.Index(index_name)
    
    lines = rag_data.splitlines('\n')

    for line in lines:
        if line.strip() != '':
            sentences.append(line.strip())

    sentences = sentences[:VECTOR_LIMIT]

    for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
         # find end of batch
        i_end = min(i+BATCH_SIZE, len(sentences))
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadata = [{'text': text} for text in sentences[i:i_end]]
        # create embeddings
        xc = model.encode(sentences[i:i_end])
        # create records list for upsert
        records = zip(ids, xc, metadata)
        # upsert to Pinecone
        index.upsert(vectors=records)

    # Progress bar initialization
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        # Get index stats
        index_stats = index.describe_index_stats()
        current_vector_count = int(index_stats.total_vector_count)

        # Calculate progress
        progress = min(current_vector_count / len(sentences), 1.0)

        # Update progress bar and status text
        progress_bar.progress(progress)
        status_text.text(f"Indexing RAG: {int(progress * 100)}%")

        # Check if indexing is complete
        if current_vector_count >= len(sentences):
            st.success("Indexing is complete!")
            break

        # Wait before polling again
        time.sleep(2)  # Adjust polling interval as needed
    

def retrieve_relevant_rag_data(query):
  
    index_name = "rag-data-index-test"
    index = pc.Index(index_name)
    
    xq = model.encode(query).tolist()
  
    # Retrieve top 3 most relevant vectors from Pinecone
    result = index.query(vector=xq, top_k=3, include_metadata=True)
    
    output = ""
    olen = min(5, len(result['matches']))
    for i in range(olen):
        text = result['matches'][i]['metadata']['text']
        output = output + str(i+1) + ")" + text
    return output


def generate_response(input_text, rag_data, openai_api_key, pinecone_api_key):
    
    st.chat_message("assistant").write(random.choice(searching_phrases))
    # Retrieve relevant RAG data for augmentation
    relevant_rag_data = retrieve_relevant_rag_data(input_text)
    if detail_toggle:
        st.info(relevant_rag_data)
    
    # Combine truncated RAG data with user input
    augmented_input = f"Context: {relevant_rag_data}\n\nUser Question: {input_text}"
    
    # Call the model with the sanitized input 
    openmodel = ChatOpenAI(
        model=LLM, 
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


with tab3:
    with st.form("rag_form"):
        rag_data = st.text_area("Enter RAG Data", "Lots of nice things about something")

        submitrag = st.form_submit_button("Submit RAG")
        
        # Validation for API keys
        if not pinecone_api_key:
            st.warning("Please enter your Pinecone API key!")
        
        if submitrag and pinecone_api_key:
            st.toast("wait for it...", icon="⏳")
            # Store RAG data in Pinecone
            store_rag_data_in_pinecone(rag_data)

with tab1:
    st.title('CatBot {^o_o^}')
    st.caption("A friendly feline powered by OpenAI and your very own RAG database")
    st.chat_message("assistant").write(f"How can I help? I'm feeling {params["grammar"]} {selected_archetype} today!")

    if prompt := st.chat_input():
        if openai_api_key.startswith("sk-") and pinecone_api_key:
            generate_response(prompt, rag_data, openai_api_key, pinecone_api_key)
