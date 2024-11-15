import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import for embeddings
from langchain_openai.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
import pinecone
import numpy as np
import time
import os
from tqdm.auto import tqdm
from pinecone import ServerlessSpec

PINECONEAPI = os.getenv("PINECONEAPI")
OPENAIKEY = os.getenv("OPENAIKEY")
LLM="gpt-3.5-turbo"

BATCH_SIZE = 128
VECTOR_LIMIT = 1024


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

st.title('Catbot {^o_o^}')
st.sidebar.image("avocado_cat_transparent.png", caption="Ask Catbot", use_container_width=True)


openai_api_key = st.sidebar.text_input("OpenAI API Key", OPENAIKEY, type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key",PINECONEAPI, type="password")
    
if pinecone_api_key:
    # Initialize Pinecone using the Pinecone class
    pc = pinecone.Pinecone(api_key=pinecone_api_key)

# Add "detail" toggle to switch between raw response and content only
detail_toggle = st.sidebar.checkbox("Detail", value=False)

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
    
    # Retrieve relevant RAG data for augmentation
    relevant_rag_data = retrieve_relevant_rag_data(input_text)
    if detail_toggle:
        st.info(relevant_rag_data)
    
    # Combine truncated RAG data with user input
    augmented_input = f"Context: {relevant_rag_data}\n\nUser Question: {input_text}"
    
    # Call the model with the sanitized input
    openmodel = ChatOpenAI(model=LLM, temperature=0.7, api_key=openai_api_key)
    response = openmodel.invoke(augmented_input)

    # Check the toggle to decide output format
    if detail_toggle:
        # Display full raw response
        st.json(response)
    else:
        # Display only the 'content' part of the response
        st.info(response.content)


with st.form("rag_form"):
    rag_data = st.sidebar.text_area("Enter RAG Data", "Lots of nice things about something")

    submitrag = st.form_submit_button("Submit RAG")
    
    # Validation for API keys
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if not pinecone_api_key:
        st.warning("Please enter your Pinecone API key!", icon="⚠")
    
    if submitrag and openai_api_key.startswith("sk-") and pinecone_api_key:
        # Store RAG data in Pinecone
        store_rag_data_in_pinecone(rag_data)

with st.form("my_form"):
    text = st.text_area(
        "Ask Catbot:",
        "Who is Steve?",
    )
    submitted = st.form_submit_button("Submit")
    
    # Validation for API keys
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if not pinecone_api_key:
        st.warning("Please enter your Pinecone API key!", icon="⚠")
    
    if submitted and openai_api_key.startswith("sk-") and pinecone_api_key:
        generate_response(text, rag_data, openai_api_key, pinecone_api_key)
