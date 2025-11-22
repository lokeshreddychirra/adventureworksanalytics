import os
import shutil
import pandas as pd
import numpy as np
import pickle
import re
from pyspark.sql.functions import current_timestamp, lit
from databricks.connect import DatabricksSession
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Configuration ---
LOCAL_DATA_PATH = os.path.abspath("rag_data")
SCHEMA_NAME = "rag_demo"

# --- Setup ---
print("Initializing Databricks Session...")
try:
    # Attempt to use serverless compute with the specified profile
    spark = DatabricksSession.builder.profile("clokeshreddy").serverless(True).getOrCreate()
    print("Session created successfully with Serverless Compute.")
except Exception as e:
    print(f"Error creating session: {e}")
    print("Please ensure your .databrickscfg has the [clokeshreddy] profile and that Serverless is enabled in your workspace.")
    exit(1)

if os.path.exists(LOCAL_DATA_PATH):
    shutil.rmtree(LOCAL_DATA_PATH)
os.makedirs(LOCAL_DATA_PATH)

print(f"Creating schema {SCHEMA_NAME}...")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")

# --- Step 1: Ingest Data ---
print("\n--- Step 1: Ingest Data ---")
sample_docs = {
    "doc1.txt": "Delta Lake is an open-source storage layer that brings ACID transactions to Apache Spark and big data workloads.",
    "doc2.txt": "Databricks is a unified data analytics platform for massive scale data engineering and data science.",
    "doc3.txt": "Retrieval-Augmented Generation (RAG) combines an LLM with a retrieval system to provide accurate, up-to-date answers.",
    "doc4.txt": "Apache Spark is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters."
}

data = []
for filename, content in sample_docs.items():
    data.append({"source_file": filename, "raw_content": content})

pdf_raw = pd.DataFrame(data)
df_raw = spark.createDataFrame(pdf_raw)
df_raw = df_raw.withColumn("ingestion_time", current_timestamp())

print("Saving to raw_documents...")
df_raw.write.format("delta").mode("overwrite").saveAsTable("raw_documents")
df_raw.show()

# --- Step 2: Generate Chunks ---
print("\n--- Step 2: Generate Chunks ---")
# Load raw data to local pandas for processing (avoiding cluster lib issues)
df_raw_read = spark.table("raw_documents")
pdf_raw_read = df_raw_read.toPandas()

def clean_text(text):
    if text is None: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

pdf_raw_read['cleaned_content'] = pdf_raw_read['raw_content'].apply(clean_text)

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

chunks_data = []
for _, row in pdf_raw_read.iterrows():
    chunks = splitter.split_text(row['cleaned_content'])
    for chunk in chunks:
        chunks_data.append({
            "source_file": row['source_file'],
            "chunk_text": chunk
        })

pdf_chunks = pd.DataFrame(chunks_data)
# Add IDs
pdf_chunks['chunk_id'] = range(len(pdf_chunks))

print("Saving to silver_chunks...")
df_chunks = spark.createDataFrame(pdf_chunks)
df_chunks.write.format("delta").mode("overwrite").saveAsTable("silver_chunks")
df_chunks.show()

# --- Step 3: Generate Embeddings ---
print("\n--- Step 3: Generate Embeddings ---")
# Load chunks locally
df_chunks_read = spark.table("silver_chunks")
pdf_chunks_read = df_chunks_read.toPandas()

model_name = "all-MiniLM-L6-v2"
print(f"Loading model {model_name}...")
model = SentenceTransformer(model_name)

print("Generating embeddings...")
embeddings = model.encode(pdf_chunks_read['chunk_text'].tolist())
pdf_chunks_read['embedding'] = embeddings.tolist()

print("Saving to gold_embeddings...")
# Spark might have trouble inferring array<float> from list, so we define schema or let it infer
# We'll try letting it infer. If it fails, we might need schema.
# Usually list of floats works.
df_embeddings = spark.createDataFrame(pdf_chunks_read)
df_embeddings.write.format("delta").mode("overwrite").saveAsTable("gold_embeddings")
df_embeddings.show()

# --- Step 4: Create Vector Index ---
print("\n--- Step 4: Create Vector Index ---")
# We already have pdf_chunks_read with embeddings locally, but let's read from table to be sure
df_gold = spark.table("gold_embeddings")
pdf_gold = df_gold.select("chunk_id", "embedding").toPandas()

embeddings_list = pdf_gold["embedding"].tolist()
embeddings_array = np.array(embeddings_list).astype("float32")
d = embeddings_array.shape[1]

print(f"Building FAISS index with dimension {d}...")
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)

index_path = os.path.join(LOCAL_DATA_PATH, "faiss_index.bin")
mapping_path = os.path.join(LOCAL_DATA_PATH, "id_mapping.pickle")

print(f"Saving index to {index_path}...")
faiss.write_index(index, index_path)

id_mapping = {i: chunk_id for i, chunk_id in enumerate(pdf_gold["chunk_id"])}
with open(mapping_path, "wb") as f:
    pickle.dump(id_mapping, f)

# --- Step 5: RAG Agent ---
print("\n--- Step 5: RAG Agent ---")
# Load resources
print("Loading LLM...")
llm_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
generator = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer, max_length=512)

# Retrieval function
def retrieve_context(query, k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    
    retrieved_ids = [id_mapping[i] for i in indices[0] if i != -1]
    if not retrieved_ids:
        return []
    
    # Fetch text from Spark (or we could use our local dataframe since it's small)
    # Using Spark to demonstrate connection
    ids_str = ",".join([str(id) for id in retrieved_ids])
    # Note: In real app, use parameterized query or filter
    # For simplicity:
    df_context = spark.sql(f"SELECT chunk_text, source_file FROM {SCHEMA_NAME}.gold_embeddings WHERE chunk_id IN ({ids_str})")
    return df_context.collect()

def rag_agent(query):
    print(f"Query: {query}")
    context_rows = retrieve_context(query)
    context_text = "\n\n".join([row.chunk_text for row in context_rows])
    
    prompt = f"""
    Answer the question based on the context below. If the answer is not in the context, say "I don't know".
    
    Context:
    {context_text}
    
    Question:
    {query}
    
    Answer:
    """
    
    response = generator(prompt)
    return response[0]['generated_text']

# Test
print("\nTest 1:")
print(rag_agent("Explain Delta Lake architecture"))

print("\nTest 2:")
print(rag_agent("What is Databricks?"))

print("\nDone!")
