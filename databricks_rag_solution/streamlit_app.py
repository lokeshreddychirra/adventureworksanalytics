import streamlit as st
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from databricks.connect import DatabricksSession

# --- Configuration ---
LOCAL_DATA_PATH = os.path.abspath("rag_data")
SCHEMA_NAME = "rag_demo"
PAGE_TITLE = "Databricks RAG Demo"
PAGE_ICON = "🤖"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- CSS for Aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #FF3621;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D92E1C;
        color: white;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #e0e0e0;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .message {
      width: 80%;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Layout (Render this FIRST) ---
st.title("🧠 Enterprise Knowledge Base")
st.markdown("Ask questions about your Databricks and Delta Lake documents.")

# --- Sidebar ---
st.sidebar.title(f"{PAGE_ICON} Settings")
st.sidebar.markdown("This app connects to your **Databricks Serverless** compute to retrieve documents and uses a local LLM to generate answers.")

# --- Initialization ---
@st.cache_resource
def get_spark_session():
    print("Connecting to Databricks...")
    try:
        return DatabricksSession.builder.profile("clokeshreddy").serverless(True).getOrCreate()
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

@st.cache_resource
def load_resources():
    print("Loading resources...")
    # Load Index
    index_path = os.path.join(LOCAL_DATA_PATH, "faiss_index.bin")
    mapping_path = os.path.join(LOCAL_DATA_PATH, "id_mapping.pickle")
    
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        return None, None, None, None, None

    index = faiss.read_index(index_path)
    with open(mapping_path, "rb") as f:
        id_mapping = pickle.load(f)
        
    # Load Models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    llm_model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
    generator = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer, max_length=512)
    print("Resources loaded.")
    return index, id_mapping, embed_model, generator

# Load with visual feedback
with st.spinner("Connecting to Databricks..."):
    spark = get_spark_session()
    if not spark:
        st.error("Failed to connect to Databricks. Check terminal for details.")
        st.stop()

with st.spinner("Loading AI Models & Index..."):
    index, id_mapping, embed_model, generator = load_resources()

if not index:
    st.warning("⚠️ Index not found! Please run the data ingestion script first or check the `rag_data` folder.")
    st.stop()

# --- Logic ---
def retrieve_context(query, k=3):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    
    retrieved_ids = [id_mapping[i] for i in indices[0] if i != -1]
    if not retrieved_ids:
        return []
    
    # Fetch text from Spark
    ids_str = ",".join([str(id) for id in retrieved_ids])
    try:
        df_context = spark.sql(f"SELECT chunk_text, source_file FROM {SCHEMA_NAME}.gold_embeddings WHERE chunk_id IN ({ids_str})")
        return df_context.collect()
    except Exception as e:
        st.error(f"Spark Query Failed: {e}")
        return []

def generate_answer(query):
    context_rows = retrieve_context(query)
    context_text = "\n\n".join([f"[{row.source_file}] {row.chunk_text}" for row in context_rows])
    
    prompt = f"""
    Answer the question based on the context below. If the answer is not in the context, say "I don't know".
    
    Context:
    {context_text}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = generator(prompt)
    return response[0]['generated_text'], context_rows

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is Delta Lake?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        answer, context_rows = generate_answer(prompt)
        
        # Format response with sources
        full_response = f"{answer}\n\n**Sources:**"
        for row in context_rows:
            full_response += f"\n- *{row.source_file}*"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(full_response)
        with st.expander("View Retrieved Context"):
            for row in context_rows:
                st.info(f"**{row.source_file}**: {row.chunk_text}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
