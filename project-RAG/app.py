"""Simple RAG Application with Streamlit UI"""
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv
import msgpack

# Load environment variables from .env file
load_dotenv()

# Configuration
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
INDEX_NAME = os.getenv("INDEX_NAME", "simple_rag")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def create_index():
    """Create vector index in Endee"""
    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/create",
            json={
                "name": INDEX_NAME,
                "dim": 384,
                "space": "cosine",
                "precision": "float32",
                "M": 16,
                "ef_construction": 200
            }
        )
        return response.ok
    except:
        return False

def chunk_text(text, chunk_size=500):
    """Split text into chunks"""
    words = text.split()
    if len(words) == 0:
        return []
    
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    # If text is shorter than chunk_size, return it as single chunk
    if len(chunks) == 0 and text.strip():
        chunks.append(text)
    
    return chunks

def ingest_document(text, model):
    """Ingest document into Endee"""
    chunks = chunk_text(text)
    
    if len(chunks) == 0:
        st.error("No chunks created from text!")
        return 0
    
    st.info(f"Created {len(chunks)} chunks from document")
    
    embeddings = model.encode(chunks, normalize_embeddings=True)
    
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Convert to float32 explicitly
        vector_data = embedding.astype(np.float32).tolist()
        vectors.append({
            "id": str(i),
            "vector": vector_data,
            "meta": chunk
        })
    
    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
            json=vectors,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.ok:
            st.success(f"Successfully inserted {len(chunks)} vectors into Endee")
            return len(chunks)
        else:
            st.error(f"Endee API error: {response.status_code} - {response.text}")
            return 0
    except Exception as e:
        st.error(f"Error inserting into Endee: {str(e)}")
        return 0

def search_similar(question, model, top_k=3):
    """Search for similar chunks"""
    query_embedding = model.encode([question])[0]
    
    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
            json={
                "vector": query_embedding.astype(np.float32).tolist(),
                "k": top_k
            },
            timeout=10
        )
        
        if response.ok:
            # Decode MessagePack response
            results = msgpack.unpackb(response.content, raw=False)
            
            contexts = []
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and "meta" in r:
                        meta = r["meta"]
                        # Convert bytes to string if needed
                        if isinstance(meta, bytes):
                            meta = meta.decode('utf-8')
                        contexts.append(meta)
                    elif isinstance(r, list) and len(r) > 2:
                        # Format: [id, distance, meta]
                        meta = r[2]
                        if isinstance(meta, bytes):
                            meta = meta.decode('utf-8')
                        contexts.append(meta)
            
            return contexts
        return []
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_answer(question, context):
    """Generate answer using Groq"""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return "⚠️ Please add your Groq API key in the .env file"
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Streamlit UI
st.title("🤖 Simple RAG with Endee")

# Check if Endee is running
try:
    response = requests.get(f"{ENDEE_URL}/api/v1/health", timeout=2)
    st.sidebar.success("✅ Endee Connected")
except:
    st.sidebar.error("❌ Endee not running! Start Endee first.")
    st.stop()

# Check API key
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    st.sidebar.success("✅ Groq API Key Loaded")
else:
    st.sidebar.warning("⚠️ Add Groq API key in .env file")

# Initialize index
if st.sidebar.button("Initialize Index"):
    if create_index():
        st.sidebar.success("Index created!")
    else:
        st.sidebar.info("Index already exists or created")

# Document upload
st.header("📄 Upload Document")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file and st.button("Ingest Document"):
    model = load_embedding_model()
    text = uploaded_file.read().decode("utf-8")
    
    with st.spinner("Processing document..."):
        num_chunks = ingest_document(text, model)
    
    st.success(f"✅ Ingested {num_chunks} chunks!")

# Query interface
st.header("💬 Ask Questions")
question = st.text_input("Enter your question:")

if question:
    model = load_embedding_model()
    
    with st.spinner("Searching..."):
        contexts = search_similar(question, model)
    
    if contexts:
        with st.spinner("Generating answer..."):
            context_text = "\n\n".join(contexts)
            answer = generate_answer(question, context_text)
        
        st.markdown("### Answer")
        st.write(answer)
        
        with st.expander("View Sources"):
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Source {i}:**")
                st.text(ctx[:200] + "...")
    else:
        st.warning("No relevant documents found")
