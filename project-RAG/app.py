"""Simple RAG app with Streamlit, Endee, and Groq."""

import os

import msgpack
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def resolve_endee_url() -> str:
    """Resolve Endee URL for local and Render deployments."""
    explicit_url = os.getenv("ENDEE_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    hostport = os.getenv("ENDEE_HOSTPORT")
    if hostport:
        if hostport.startswith("http://") or hostport.startswith("https://"):
            return hostport.rstrip("/")
        return f"http://{hostport}".rstrip("/")

    return "http://localhost:8080"


ENDEE_URL = resolve_endee_url()
INDEX_NAME = os.getenv("INDEX_NAME", "simple_rag")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def endee_headers(content_type_json: bool = False):
    """Build request headers for Endee API calls."""
    headers = {}
    if ENDEE_AUTH_TOKEN:
        headers["Authorization"] = ENDEE_AUTH_TOKEN
    if content_type_json:
        headers["Content-Type"] = "application/json"
    return headers or None


def is_endee_available() -> bool:
    """Check Endee health endpoint."""
    try:
        response = requests.get(
            f"{ENDEE_URL}/api/v1/health",
            headers=endee_headers(),
            timeout=2,
        )
        return response.ok
    except Exception:
        return False


@st.cache_resource
def load_embedding_model():
    """Lazy-load embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def create_index() -> bool:
    """Create vector index in Endee."""
    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/create",
            json={
                "name": INDEX_NAME,
                "dim": 384,
                "space": "cosine",
                "precision": "float32",
                "M": 16,
                "ef_construction": 200,
            },
            headers=endee_headers(content_type_json=True),
            timeout=10,
        )
        return response.ok
    except Exception:
        return False


def chunk_text(text: str, chunk_size: int = 500):
    """Split text into chunks by word count."""
    words = text.split()
    if not words:
        return []

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_document(text: str, model) -> int:
    """Ingest document chunks into Endee."""
    chunks = chunk_text(text)
    if not chunks:
        st.error("No chunks created from text.")
        return 0

    st.info(f"Created {len(chunks)} chunks from document")
    embeddings = model.encode(chunks, normalize_embeddings=True)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": str(i),
                "vector": embedding.astype(np.float32).tolist(),
                "meta": chunk,
            }
        )

    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
            json=vectors,
            headers=endee_headers(content_type_json=True),
            timeout=30,
        )

        if response.ok:
            st.success(f"Inserted {len(chunks)} vectors into Endee")
            return len(chunks)

        st.error(f"Endee API error: {response.status_code} - {response.text}")
        return 0
    except Exception as exc:
        st.error(f"Error inserting into Endee: {exc}")
        return 0


def search_similar(question: str, model, top_k: int = 3):
    """Search Endee for relevant chunks."""
    query_embedding = model.encode([question])[0]

    try:
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
            json={"vector": query_embedding.astype(np.float32).tolist(), "k": top_k},
            headers=endee_headers(content_type_json=True),
            timeout=10,
        )

        if not response.ok:
            return []

        results = msgpack.unpackb(response.content, raw=False)
        contexts = []

        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict) and "meta" in item:
                    meta = item["meta"]
                elif isinstance(item, list) and len(item) > 2:
                    meta = item[2]
                else:
                    continue

                if isinstance(meta, bytes):
                    meta = meta.decode("utf-8", errors="replace")
                contexts.append(meta)

        return contexts
    except Exception as exc:
        st.error(f"Search error: {exc}")
        return []


def generate_answer(question: str, context: str) -> str:
    """Generate answer using Groq."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return "Please add GROQ_API_KEY."

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
        max_tokens=500,
    )

    return response.choices[0].message.content


st.title("Simple RAG with Endee")

endee_available = is_endee_available()
if endee_available:
    st.sidebar.success("Endee connected")
else:
    st.sidebar.error("Endee unavailable")
    st.info("Vector DB is unreachable right now. Retry after Endee wakes up.")

if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    st.sidebar.success("Groq key loaded")
else:
    st.sidebar.warning("Set GROQ_API_KEY")

if st.sidebar.button("Initialize Index", disabled=not endee_available):
    if create_index():
        st.sidebar.success("Index created")
    else:
        st.sidebar.info("Index already exists or request failed")

st.header("Upload Document")
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file and st.button("Ingest Document", disabled=not endee_available):
    model = load_embedding_model()
    text = uploaded_file.read().decode("utf-8", errors="replace")

    with st.spinner("Processing document..."):
        num_chunks = ingest_document(text, model)

    st.success(f"Ingested {num_chunks} chunks")

st.header("Ask Questions")
question = st.text_input("Enter your question:")

if question:
    if not endee_available:
        st.warning("Endee is unavailable, so search cannot run yet.")
        st.stop()

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
